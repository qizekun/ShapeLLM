import os
import multiprocessing


def main(i):
    os.system(f'CUDA_VISIBLE_DEVICES={i} bash ReConV2/scripts/cls.sh {i} test{i} ReConV2/ckpt-last.pth')


if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=8)
    for i in range(8):
        p = pool.apply_async(main, (i,))
    pool.close()
    pool.join()

