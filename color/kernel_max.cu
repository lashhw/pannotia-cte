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
 * the U.S. Export Administration Regulations ("EAR"�) (15 C.F.R Sections 730-774),  *
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


/**
 * @brief   color kernel 1
 * @param   row         CSR pointer array
 * @param   col         CSR column array
 * @param   node_value  Vertex value array
 * @param   color_array Color value array
 * @param   stop        Termination variable
 * @param   max_d       Max array
 * @param   color       Current color label
 * @param   num_nodes   Number of vertices
 * @param   num_edges   Number of edges
 */
__global__ void color1(int *row, int *col, int *node_value, int *color_array,
                       int *stop, int *max_d, const int color,
                       const int num_nodes, const int num_edges)
{
    // Get my thread workitem id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        // If the vertex is still not colored
        if (color_array[tid] == -1) {

            // Get the start and end pointer of the neighbor list
            int start = row[tid];
            int end;
            if (tid + 1 < num_nodes)
                end = row[tid + 1];
            else
                end = num_edges;

            int maximum = -1;
            // Navigate the neighbor list
            for (int edge = start; edge < end; edge++) {
                // Determine if the vertex value is the maximum in the neighborhood
                if (color_array[col[edge]] == -1 && start != end - 1) {
                    *stop = 1;
                    if (node_value[col[edge]] > maximum)
                        maximum = node_value[col[edge]];
                }
            }
            // Assign maximum the max array
            max_d[tid] = maximum;
        }
    }
}


/**
 * @brief   color kernel 2
 * @param   node_value  Vertex value array
 * @param   color_array Color value array
 * @param   max_d       Max array
 * @param   color       Current color label
 * @param   num_nodes   Number of vertices
 * @param   num_edges   Number of edges
 */
__global__ void color2(int *node_value, int *color_array, int *max_d,
                       const int color, const int num_nodes,
                       const int num_edges)
{
    // Get my workitem id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        // If the vertex is still not colored
        if (color_array[tid] == -1) {
            if (node_value[tid] >= max_d[tid])
                // Assign a color
                color_array[tid] = color;
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
__global__ void color1_cte(const int* __restrict__ row,
                           const int* __restrict__ col,
                           const int* __restrict__ node_value,
                           const int* __restrict__ color_array,
                           int* __restrict__ stop,
                           int* __restrict__ max_d,
                           const int color,
                           const int num_nodes,
                           const int num_edges) {
    // statically assert that THREADS_PER_BLOCK must be a multiple of WARP_SIZE
    static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be a multiple of WARP_SIZE");

    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    volatile __shared__ int scans[WARPS_PER_BLOCK][WARP_SIZE];
    volatile __shared__ int mapped[WARPS_PER_BLOCK][WARP_SIZE];
    volatile __shared__ int reds[WARPS_PER_BLOCK][WARP_SIZE];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    scans[warp_id][lane_id] = (global_thread_id + 1 < num_nodes) ? row[global_thread_id + 1] : num_edges;
    int global_fine_task_start_id = (global_thread_id - lane_id < num_nodes) ? row[global_thread_id - lane_id] : num_edges;

    reds[warp_id][lane_id] = -1;

    int num_fine_tasks = scans[warp_id][WARP_SIZE - 1] - global_fine_task_start_id;

    for (int fine_task_id = lane_id;
         fine_task_id < num_fine_tasks;
         fine_task_id += WARP_SIZE) {
        int global_fine_task_id = global_fine_task_start_id + fine_task_id;

        int coarse_task_id = binary_search(scans[warp_id], global_fine_task_id, 0, WARP_SIZE);
        int global_coarse_task_id = global_thread_id - lane_id + coarse_task_id;
        if (color_array[global_coarse_task_id] == -1) {
            int global_coarse_task_fine_start_id = (coarse_task_id == 0) ? global_fine_task_start_id : scans[warp_id][coarse_task_id - 1];
            int global_coarse_task_fine_end_id = scans[warp_id][coarse_task_id];
            int seg_start_id = global_coarse_task_fine_start_id - global_fine_task_start_id;
            int in_seg_id = min(lane_id, fine_task_id - seg_start_id);
            int seg_size = min(global_coarse_task_fine_end_id - global_fine_task_id, WARP_SIZE - lane_id) + in_seg_id;

            mapped[warp_id][lane_id] = -1;
            if (color_array[col[global_fine_task_id]] == -1 &&
                global_coarse_task_fine_start_id != global_coarse_task_fine_end_id - 1) {
                *stop = 1;
                mapped[warp_id][lane_id] = node_value[col[global_fine_task_id]];
            }

            for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
                if (in_seg_id + stride < seg_size) {
                    mapped[warp_id][lane_id] = max(mapped[warp_id][lane_id], mapped[warp_id][lane_id + stride]);
                }
            }

            if (in_seg_id == 0)
                reds[warp_id][coarse_task_id] = max(reds[warp_id][coarse_task_id], mapped[warp_id][lane_id]);
        }
    }

    if (color_array[global_thread_id] == -1 && global_thread_id < num_nodes) max_d[global_thread_id] = reds[warp_id][lane_id];
}
