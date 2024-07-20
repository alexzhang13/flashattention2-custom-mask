import triton

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"
