python python/spinn/data/arithmetic/make_data.py \
--data_type simple \
--length 7 \
--limit 800 \
--balance_length \
> ./python/spinn/data/arithmetic/simple7_800.tsv

python python/spinn/data/arithmetic/make_data.py \
--data_type simple \
--limit 100000 \
--length 7 \
--exclude ./python/spinn/data/arithmetic/simple7_800.tsv \
> ./python/spinn/data/arithmetic/simple7_100k.tsv
