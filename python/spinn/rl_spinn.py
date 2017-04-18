import itertools
import copy

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import LSTMState, Embed, MLP
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, the_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example

from spinn.fat_stack import BaseModel as _BaseModel
from spinn.fat_stack import SPINN

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS, context_args, composition_args):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         mlp_dim=FLAGS.mlp_dim,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
         transition_weight=FLAGS.transition_weight,
         use_sentence_pair=use_sentence_pair,
         lateral_tracking=FLAGS.lateral_tracking,
         use_tracking_in_composition=FLAGS.use_tracking_in_composition,
         predict_use_cell=FLAGS.predict_use_cell,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
         rl_mu=FLAGS.rl_mu,
         rl_epsilon=FLAGS.rl_epsilon,
         rl_baseline=FLAGS.rl_baseline,
         rl_reward=FLAGS.rl_reward,
         rl_weight=FLAGS.rl_weight,
         rl_whiten=FLAGS.rl_whiten,
         rl_valid=FLAGS.rl_valid,
         rl_entropy=FLAGS.rl_entropy,
         rl_entropy_beta=FLAGS.rl_entropy_beta,
         rl_catalan=FLAGS.rl_catalan,
         context_args=context_args,
         composition_args=composition_args,
        )


class RLSPINN(SPINN):
    epsilon = 1.0
    temperature = 1.0
    catalan = True

    def predict_actions(self, transition_output):
        transition_dist = F.softmax(transition_output / max(self.temperature, 1e-8)).data.cpu()

        if self.training:
            # Interpolate between the uniform random distrubition of binary trees
            # and the distribution from the transition_net's softmax.
            if self.catalan:
                A = F.softmax(transition_output).data[:,0].cpu()
                B = torch.zeros(A.size()).fill_(0.5)
                p = transition_dist[:,0]
                i = (p-B)/(A-B)
                C = [self.shift_probabilities.prob(n_red, n_step, n_tok)
                    for n_red, n_step, n_tok in zip(self.n_reduces, self.n_steps, self.n_tokens)]
                C = torch.FloatTensor(C).unsqueeze(1)
                transition_dist = torch.cat([C, 1-C], 1)

            transitions_sampled = torch.multinomial(transition_dist, 1).view(-1).numpy()
            transition_preds = transitions_sampled
        else:
            transition_greedy = transition_dist.cpu().numpy().argmax(axis=1)
            transition_preds = transition_greedy
        return transition_preds


class BaseModel(_BaseModel):

    optimize_transition_loss = False

    def __init__(self,
                 rl_mu=None,
                 rl_baseline=None,
                 rl_reward=None,
                 rl_weight=None,
                 rl_whiten=None,
                 rl_valid=None,
                 rl_epsilon=None,
                 rl_entropy=None,
                 rl_entropy_beta=None,
                 rl_catalan=None,
                 **kwargs):
        super(BaseModel, self).__init__(**kwargs)

        self.kwargs = kwargs

        self.rl_mu = rl_mu
        self.rl_baseline = rl_baseline
        self.rl_reward = rl_reward
        self.rl_weight = rl_weight
        self.rl_whiten = rl_whiten
        self.rl_valid = rl_valid
        self.rl_entropy = rl_entropy
        self.rl_entropy_beta = rl_entropy_beta
        self.spinn.epsilon = rl_epsilon
        self.spinn.catalan = rl_catalan

        if self.rl_baseline == "value":
            self.v_dim = 100
            self.v_rnn = nn.LSTM(self.input_dim, self.v_dim, num_layers=1, batch_first=True)
            self.v_mlp = MLP(self.v_dim,
                mlp_dim=1024, num_classes=1, num_mlp_layers=2,
                mlp_bn=True, classifier_dropout_rate=0.1)

        self.register_buffer('baseline', torch.FloatTensor([0.0]))

    def build_spinn(self, args, vocab, predict_use_cell):
        return RLSPINN(args, vocab, predict_use_cell)

    def forward_hook(self, embeds, batch_size, seq_length):
        if self.rl_baseline == "value" and self.training:
            # Break the computational graph.
            x = Variable(embeds.data, volatile=not self.training).view(batch_size, seq_length, -1)
            h0 = Variable(to_gpu(torch.zeros(1, batch_size, self.v_dim)), volatile=not self.training)
            c0 = Variable(to_gpu(torch.zeros(1, batch_size, self.v_dim)), volatile=not self.training)
            output, (hn, cn) = self.v_rnn(x, (h0, c0))
            self.baseline_outp = self.v_mlp(hn.squeeze())

    def run_greedy(self, sentences, transitions):
        inference_model_cls = BaseModel

        # HACK: This is a pretty simple way to create the inference time version of SPINN.
        # The reason a copy is necessary is because there is some retained state in the
        # memories and loss variables that break deep copy.
        inference_model = inference_model_cls(**self.kwargs)
        inference_model.load_state_dict(copy.deepcopy(self.state_dict()))
        inference_model.eval()

        if the_gpu.gpu >= 0:
            inference_model.cuda()
        else:
            inference_model.cpu()

        outputs = inference_model(sentences, transitions,
            use_internal_parser=True,
            validate_transitions=True)

        return outputs

    def build_reward(self, probs, target, rl_reward="standard"):
        if rl_reward == "standard": # Zero One Loss.
            rewards = torch.eq(probs.max(1)[1], target).float()
        elif rl_reward == "xent": # Cross Entropy Loss.
            _target = target.long().view(-1, 1)
            log_inv_prob = torch.log(1 - probs) # get the log of the inverse probabilities
            rewards = -1 * torch.gather(log_inv_prob, 1, _target)
        else:
            raise NotImplementedError

        return rewards

    def build_baseline(self, rewards, sentences, transitions, y_batch=None, embeds=None):
        if self.rl_baseline == "ema":
            mu = self.rl_mu
            baseline = self.baseline[0]
            self.baseline[0] = self.baseline[0] * (1 - mu) + rewards.mean() * mu
        elif self.rl_baseline == "pass":
            baseline = 0.
        elif self.rl_baseline == "greedy":
            # Pass inputs to Greedy Max
            output = self.run_greedy(sentences, transitions)

            # Estimate Reward
            probs = F.softmax(output).data.cpu()
            target = torch.from_numpy(y_batch).long()
            approx_rewards = self.build_reward(probs, target, rl_reward=self.rl_reward)

            baseline = approx_rewards
        elif self.rl_baseline == "value":
            output = self.baseline_outp

            if self.rl_reward == "standard":
                baseline = F.sigmoid(output)
                self.value_loss = nn.BCELoss()(baseline, to_gpu(Variable(rewards, volatile=not self.training)))
            elif self.rl_reward == "xent":
                baseline = output
                self.value_loss = nn.MSELoss()(baseline, to_gpu(Variable(rewards, volatile=not self.training)))
            else:
                raise NotImplementedError

            baseline = baseline.data.cpu()
        else:
            raise NotImplementedError

        return baseline

    def reinforce(self, advantage):
        """
        t_preds  = 200...111 (flattened predictions from sub_batches 1...N)
        t_mask   = 011...111 (binary mask, selecting non-skips only)
        t_logits = (B*N)xC (tensor of sub_batch_size * sub_num_batches x transition classes)
        a_index  = 011...(N-1)(N-1)(N-1) (masked sub_batch_indices for each transition)
        t_index  = 013...(B*N-3)(B*N-2)(B*N-1) (masked indices across all sub_batches)
        """

        # TODO: Many of these ops are on the cpu. Might be worth shifting to GPU.

        t_preds = np.concatenate([m['t_preds'] for m in self.spinn.memories if m.get('t_preds', None) is not None])
        t_mask = np.concatenate([m['t_mask'] for m in self.spinn.memories if m.get('t_mask', None) is not None])
        t_valid_mask = np.concatenate([m['t_valid_mask'] for m in self.spinn.memories if m.get('t_mask', None) is not None])
        t_logits = torch.cat([m['t_logits'] for m in self.spinn.memories if m.get('t_logits', None) is not None], 0)

        if self.rl_valid:
            t_mask = np.logical_and(t_mask, t_valid_mask)

        batch_size = advantage.size(0)
        seq_length = t_preds.shape[0] / batch_size

        a_index = np.arange(batch_size)
        a_index = a_index.reshape(1,-1).repeat(seq_length, axis=0).flatten()
        a_index = torch.from_numpy(a_index[t_mask]).long()

        t_index = to_gpu(Variable(torch.from_numpy(np.arange(t_mask.shape[0])[t_mask])).long())

        self.stats = dict(
            mean=advantage.mean(),
            mean_magnitude=advantage.abs().mean(),
            var=advantage.var(),
            var_magnitude=advantage.abs().var()
            )

        if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two sentences.
            advantage = torch.cat([advantage, advantage], 0)

        # Expand advantage.
        advantage = torch.index_select(advantage, 0, a_index)

        # Filter logits.
        t_logits = torch.index_select(t_logits, 0, t_index)

        actions = to_gpu(Variable(torch.from_numpy(t_preds[t_mask]).long().view(-1, 1), volatile=not self.training))
        log_p_action = torch.gather(t_logits, 1, actions)

        # source: https://github.com/miyosuda/async_deep_reinforce/issues/1
        if self.rl_entropy:
            # TODO: Taking exp of a log is not the best way to get the initial probability...
            entropy = - (t_logits * torch.exp(t_logits)).sum(1)
        else:
            entropy = 0.0

        policy_loss = -1. * torch.sum(log_p_action * to_gpu(Variable(advantage, volatile=log_p_action.volatile)) + entropy * self.rl_entropy_beta)
        policy_loss /= log_p_action.size(0)
        policy_loss *= self.rl_weight

        return policy_loss

    def output_hook(self, output, sentences, transitions, y_batch=None, embeds=None):
        if not self.training:
            return

        probs = F.softmax(output).data.cpu()
        target = torch.from_numpy(y_batch).long()

        # Get Reward.
        rewards = self.build_reward(probs, target, rl_reward=self.rl_reward)

        # Get Baseline.
        baseline = self.build_baseline(rewards, sentences, transitions, y_batch, embeds)

        # Calculate advantage.
        advantage = rewards - baseline

        # Whiten advantage. This is also called Variance Normalization.
        if self.rl_whiten:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Assign REINFORCE output.
        self.policy_loss = self.reinforce(advantage)
