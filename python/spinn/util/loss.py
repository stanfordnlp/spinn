def auxiliary_loss(model):

    has_spinn = hasattr(model, 'spinn')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')

    # Optionally calculate transition loss/accuracy.
    policy_loss = model.policy_loss if has_policy else None
    value_loss = model.value_loss if has_value else None

    total_loss = 0.0
    if has_policy:
        total_loss += model.policy_loss
    if has_value:
        total_loss += model.value_loss

    return total_loss