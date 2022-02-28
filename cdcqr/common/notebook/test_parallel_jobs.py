from cdcqr.common.utils import parallel_jobs

if __name__ == "__main__":
    def f(x):
        return x*2
    
    arg_list = list(range(1,5000))
    
    ret = parallel_jobs(f, arg_list)
    
    print(ret[1])
    
