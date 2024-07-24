import cs336_basics

import argparse
import os
import sys
import timeit

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

import cs336_basics.model
import cs336_basics.optimizer

def profile_naive(model, args, tokens, optimizer):
    # Train the model
    # warm up
    if args.warm_up:
        loss = model(tokens).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # forward
    forward_times = []
    for _ in range(5):
        start = timeit.default_timer()
        model(tokens)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)
    forward_time_mean = sum(forward_times) / len(forward_times)
    forward_time_std = torch.tensor(forward_times).std().item()
    print("Forward time (mean): {:.2f} seconds".format(forward_time_mean), 
            "std: {:.2f}".format(forward_time_std))

    # backward
    backward_times = []
    for _ in range(5):
        loss = model(tokens).mean()
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        backward_times.append(end - start)
    backward_time_mean = sum(backward_times) / len(backward_times)
    backward_time_std = torch.tensor(backward_times).std().item()
    print("Backward time (mean): {:.2f} seconds".format(backward_time_mean), 
            "std: {:.2f}".format(backward_time_std))
    return [forward_time_mean, forward_time_std, backward_time_mean, backward_time_std]

def run_step(model, inputs, optimizer, enable_backward):
    with record_function('forward_pass'):
        loss = model.forward(inputs).mean()
    if enable_backward:
        with record_function('backward_pass'):
            loss.backward()
        with record_function('optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

def profile_torch(model, args, tokens, optimizer):
    # Train the model
    # warm up
    run_step(model, tokens, optimizer, enable_backward=True)
    # forward
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            ], experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        ) as prof:
        for _ in range(args.n_steps):
            run_step(model, tokens, optimizer, enable_backward=True)
            prof.step()

    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

def main():
    parser = argparse.ArgumentParser(description='Profile the model')
    parser.add_argument('--d_model', type=int, required=True, help='The dimension of the model')
    parser.add_argument('--d_ff', type=int, required=True, help='The dimension of the feedforward layer')
    parser.add_argument('--num_layers', type=int, required=True, help='The number of layers')
    parser.add_argument('--num_heads', type=int, required=True, help='The number of heads')
    parser.add_argument('--warm_up', action='store_true', help='Whether to warm up the model')
    parser.add_argument('--n_steps', type=int, default=5, help='The number of steps to profile')
    args = parser.parse_args()

    print(f"d_model: {args.d_model}")
    print(f"d_ff: {args.d_ff}")
    print(f"num_layers: {args.num_layers}")
    print(f"num_heads: {args.num_heads}")
    print(f"warm_up: {args.warm_up}")

    vocab_size = 10000
    context_length = 128
    batch_size = 16

    model = cs336_basics.model.BasicsTransformerLM(
        vocab_size=vocab_size, 
        context_length=context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )

    tokens = torch.randint(vocab_size, (batch_size, context_length))

    model = model.cuda()
    tokens = tokens.cuda()
    optimizer = cs336_basics.optimizer.AdamW(model.parameters())

    # res = profile_naive(model, args, tokens, optimizer)
    # # write to csv
    # with open("profile.csv", "a") as f:
    #     f.write(f"{args.d_model},{args.d_ff},{args.num_layers},{args.num_heads},{res[0]},{res[1]},{res[2]},{res[3]}\n")

    profile_torch(model, args, tokens, optimizer)

if __name__ == '__main__':
    main()