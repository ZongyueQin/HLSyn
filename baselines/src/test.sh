# run on 2nd instance of u5
for kernel in gemm-ncubed # stencil
do
    python3 run_tool_dse.py --kernel $kernel --benchmark machsuite --root-dir ../ --redis-port "6379"
done
