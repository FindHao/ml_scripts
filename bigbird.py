
import numpy as np
import time
import random


def _bigbird_block_rand_mask(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last: 
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn



def _bigbird_block_rand_mask_3(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.choice(middle_seq[2:last], 3)
            elif i == 2:
                rand_attn[i - 1, :] = np.random.choice(middle_seq[3:last], 3)
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.choice(middle_seq[:last], 3)
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.choice(middle_seq[:last], 3)
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.choice(middle_seq[:start], 3)
                elif (end + 1) == last: 
                    rand_attn[i - 1, :] = np.random.choice(middle_seq[:start], 3)
                else:
                    rand_attn[i - 1, :] = np.random.choice(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last])), 3
                    )
        return rand_attn



def _bigbird_block_rand_mask_4(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        middle_seq = list(middle_seq)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = random.sample(middle_seq[2:last], 3)
            elif i == 2:
                rand_attn[i - 1, :] = random.sample(middle_seq[3:last], 3)
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = random.sample(middle_seq[:last], 3)
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = random.sample(middle_seq[:last], 3)
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = random.sample(middle_seq[:start], 3)
                elif (end + 1) == last: 
                    rand_attn[i - 1, :] = random.sample(middle_seq[:start], 3)
                else:
                    rand_attn[i - 1, :] = random.sample(list(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))), 3
                    )
        return rand_attn

from itertools import permutations
def generate_one_table(o_table, start_i, end_i):
    return list(permutations(o_table[start_i:end_i], 3))


to_seq_length=4096
to_block_size=64
from_seq_length=4096
from_block_size=64
all_tables = [ [] for _ in range(64) ]

def generate_all_table():
    global all_tables
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = 15
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            # all_tables[i - 1] = np.random.permutation(middle_seq[2:last])[:r]
            all_tables[i-1] = generate_one_table(middle_seq, 2, last)
        elif i == 2:
            # rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            all_tables[i-1] = generate_one_table(middle_seq, 3, last)
        elif i == from_seq_length // from_block_size - 3:
            # rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            all_tables[i-1] = generate_one_table(middle_seq, 0, last)
        # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            # rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            all_tables[i-1] = generate_one_table(middle_seq, 0, last)
        # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                # rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                all_tables[i-1] = generate_one_table(middle_seq, 0, start)
            elif (end + 1) == last:
                # rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                all_tables[i-1] = generate_one_table(middle_seq, 0, start)
            else:
                # rand_attn[i - 1, :] = np.random.permutation(
                #     np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                # )[:r]
                new_middle_seq = np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                all_tables[i-1] = generate_one_table(new_middle_seq, 0, len(new_middle_seq))


def _bigbird_block_rand_mask2(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
    pass   
    rand_attn = [ [] for _ in range(63) ]
    n0_14 = np.random.randint(1716, size=2)
    rand_attn[0] = all_tables[0][n0_14[0]]
    one_13 = np.random.randint(1320, size=13)
    for x in range(1, 14):
        rand_attn[x] = all_tables[x][one_13[x-1]]
    rand_attn[14] = all_tables[14][n0_14[1]]
    rand_attn[15] = all_tables[15][np.random.randint(2184)]
    # n16_61 = np.random.randint(2730, size=46)
    # for x in range(16, 62):
    #     rand_attn[x] = all_tables[x][n16_61[x-16]]
    # return rand_attn

    for x in range(16, 62):
        index = np.random.randint(2730)
        rand_attn[x] = all_tables[x][index]
    return rand_attn

t0 = time.time_ns()
for i in range(432):
    
    a=_bigbird_block_rand_mask(4096, 4096, 64, 64, 3, 1024)
    if i == 1:
        print(a)
t1 = time.time_ns()
print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')




t0 = time.time_ns()
# for i in range(100):
    # a=_bigbird_block_rand_mask2(4096, 4096, 64, 64, 3, 1024)
    # if i == 1:
    #     print(a)

generate_all_table()
# for i,node in enumerate(all_tables):
#     print(i, len(node))
t1 = time.time_ns()
for i in range(432):
    a = _bigbird_block_rand_mask2(4096, 4096, 64, 64, 3, 1024)
    if i == 1:
        print(a)



print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')





# t0 = time.time_ns()
# for i in range(100):
#     a=np.random.randint(1, 63, (64,3), dtype=np.int32)
#     if i==1:
#         print(a)
# t1 = time.time_ns()
# print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')


t0 = time.time_ns()
for i in range(432):
    
    a=_bigbird_block_rand_mask_3(4096, 4096, 64, 64, 3, 1024)
    if i == 1:
        print(a)
t1 = time.time_ns()
print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')


t0 = time.time_ns()
for i in range(432):
    
    a=_bigbird_block_rand_mask_4(4096, 4096, 64, 64, 3, 1024)
    if i == 1:
        print(a)
t1 = time.time_ns()
print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')