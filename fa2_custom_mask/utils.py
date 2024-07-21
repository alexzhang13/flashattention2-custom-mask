import triton

def is_hip():
    return False
    # bugged in older versions of Triton
    # return triton.runtime.driver.active.get_current_target().backend == "hip"


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True