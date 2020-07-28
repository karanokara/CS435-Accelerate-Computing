/*

Chapter 5 in book: Global Memory (DRAM) Bandwidth

- effectively move data from the global memory into shared memories and registers

- Memory Coalescing
    - consecutive access made by threads in a warp, to global memory locations
    - hardware detect it
    - combines, or coalesces, all these accesses into One access to consecutive DRAM locations.
    - allows the DRAMs to supply data at a high rate.
    - such as:
        - Thread0 in warp 0 access N
        - Thread1 in warp 0 access N+1
        - Thread2 in warp 0 access N+2
        ...
        ...














*/