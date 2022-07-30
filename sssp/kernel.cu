/************************************************************************************\
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#define BIG_NUM 99999999

/**
 * @brief   min.+
 * @param   num_nodes  Number of vertices
 * @param   row        CSR pointer array
 * @param   col        CSR column array
 * @param   data       Weight array
 * @param   x          Input vector
 * @param   y          Output vector
 */
__global__ void
spmv_min_dot_plus_kernel(const int num_rows, int *row, int *col, int *data,
                         int *x, int *y)
{
    // Get my workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_rows) {
        // Get the start and end pointers
        int row_start = row[tid];
        int row_end = row[tid + 1];

        // Perform + for each pair of elements and a reduction with min
        int min = x[tid];
        for (int i = row_start; i < row_end; i++) {
            if (data[i] + x[col[i]] < min) {
                min = data[i] + x[col[i]];
            }
        }
        y[tid] = min;
    }
}

/**
 * @brief   min.+
 * @param   num_nodes  number of vertices
 * @param   height     the height of the adjacency matrix (col-major)
 * @param   col        the col array
 * @param   data       the data array
 * @param   x          the input vector
 * @param   y          the output vector
 */
__global__ void
ell_min_dot_plus_kernel(const int num_nodes, const int height, int *col,
                        int *data, int *x, int *y)
{
    // Get workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        int mat_offset = tid;
        int min = x[tid];

        // The vertices process a row of matrix (col-major)
        for (int i = 0; i < height; i++) {
            int mat_elem = data[mat_offset];
            int vec_elem = x[col[mat_offset]];
            if (mat_elem + vec_elem < min) {
                min = mat_elem + vec_elem;
            }
            mat_offset += num_nodes;
        }
        y[tid] = min;
    }
}

/**
 * @brief   vector_init
 * @param   vector1      vector1
 * @param   vector2      vector2
 * @param   i            source vertex id
 * @param   num_nodes    number of vertices
 */
__global__ void
vector_init(int *vector1, int *vector2, const int i, const int num_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        if (tid == i) {
            // If it is the source vertex
            vector1[tid] = 0;
            vector2[tid] = 0;
        } else {
            // If it a non-source vertex
            vector1[tid] = BIG_NUM;
            vector2[tid] = BIG_NUM;
        }
    }
}

/**
 * @brief   vector_assign
 * @param   vector1      vector1
 * @param   vector2      vector2
 * @param   num_nodes    number of vertices
 */
__global__ void
vector_assign(int *vector1, int *vector2, const int num_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        vector1[tid] = vector2[tid];
    }
}

/**
 * @brief   vector_diff
 * @param   vector1      vector1
 * @param   vector2      vector2
 * @param   stop         termination variable
 * @param   num_nodes    number of vertices
 */
__global__ void
vector_diff(int *vector1, int *vector2, int *stop, const int num_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        if (vector2[tid] != vector1[tid]) {
            *stop = 1;
        }
    }
}

__device__ __forceinline__ int binary_search(volatile int* arr, int value, int left, int right) {
    // find the smallest index in [left, right) where arr[index] > value
    // (assume we can always find a solution)

    while (left < right) {
        int mid = (left + right) >> 1;
        if (arr[mid] <= value) left = mid + 1;
        else right = mid;
    }

    return left;
}

template <int BLOCK_SIZE, int WARP_SIZE = 32>
__global__ void spmv_min_dot_plus_kernel_cte(const int num_rows,
                                             const int* __restrict__ row,
                                             const int* __restrict__ col,
                                             const int* __restrict__ data,
                                             const int* __restrict__ x,
                                             int* __restrict__ y) {
    // statically assert that THREADS_PER_BLOCK must be a multiple of WARP_SIZE
    static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be a multiple of WARP_SIZE");

    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    volatile __shared__ int scans[WARPS_PER_BLOCK][WARP_SIZE];
    volatile __shared__ int mapped[WARPS_PER_BLOCK][WARP_SIZE];
    volatile __shared__ int reds[WARPS_PER_BLOCK][WARP_SIZE];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    scans[warp_id][lane_id] = (global_thread_id < num_rows) ? row[global_thread_id + 1] : row[num_rows];
    int global_fine_task_start_id = row[global_thread_id - lane_id];

    reds[warp_id][lane_id] = BIG_NUM;

    int num_fine_tasks = scans[warp_id][WARP_SIZE - 1] - global_fine_task_start_id;

    for (int fine_task_id = lane_id;
         fine_task_id < num_fine_tasks;
         fine_task_id += WARP_SIZE) {
        int global_fine_task_id = global_fine_task_start_id + fine_task_id;

        int coarse_task_id = binary_search(scans[warp_id], global_fine_task_id, 0, WARP_SIZE);
        int global_coarse_task_fine_start_id = (coarse_task_id == 0) ? global_fine_task_start_id : scans[warp_id][coarse_task_id - 1];
        int global_coarse_task_fine_end_id = scans[warp_id][coarse_task_id];
        int seg_start_id = global_coarse_task_fine_start_id - global_fine_task_start_id;
        int in_seg_id = min(lane_id, fine_task_id - seg_start_id);
        int seg_size = min(global_coarse_task_fine_end_id - global_fine_task_id, WARP_SIZE - lane_id) + in_seg_id;

        mapped[warp_id][lane_id] = data[global_fine_task_id] + x[col[global_fine_task_id]];

        for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (in_seg_id + stride < seg_size) {
                mapped[warp_id][lane_id] = min(mapped[warp_id][lane_id], mapped[warp_id][lane_id + stride]);
            }
        }

        if (in_seg_id == 0) reds[warp_id][coarse_task_id] = min(reds[warp_id][coarse_task_id], mapped[warp_id][lane_id]);
    }

    if (global_thread_id < num_rows) y[global_thread_id] = min(x[global_thread_id], reds[warp_id][lane_id]);
}

