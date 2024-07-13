#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

// use constant memory for difficulty_5_zeros
__constant__ BYTE d_difficulty_5_zeros[SHA256_HASH_SIZE] = "0000099999999999999999999999999999999999999999999999999999999999";

// use global memory for nonce, so that it is shared between all blocks
__device__ uint64_t global_nonce = 0;

// function that searches for all nonces from 1 through MAX_NONCE (inclusive) using CUDA Threads
__global__ void findNonce(BYTE *block_content, size_t current_length, BYTE *block_hash) {
        // get global index of cuda thread
        uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

        // verify if the value of index is bigger than MAX_NONCE, otherwise contiue searching
        // for a valid nonce
        // if the nonce was already found, the other threads stop searching for a valid nonce
        if (index > (uint64_t) MAX_NONCE || global_nonce != 0)
                return;

        // create local copies for block_content and block_hash
        BYTE local_block_content[BLOCK_SIZE];
        BYTE local_block_hash[SHA256_HASH_SIZE];

        // initialize copies with the parameters of the kernel
        d_strcpy((char*)local_block_content, (const char*)block_content);
        d_strcpy((char*)local_block_hash, (const char*)block_hash);

        // make the nonce into a string, so that it can be concatenated to the local_block_content
        char nonce_string[NONCE_SIZE];
        int len_nonce = intToString(index, nonce_string);

        // concatenate the nonce string to the local_block_content
        d_strcpy((char*)(local_block_content + current_length), (const char*)nonce_string);

        // apply sha256 once and save it in local_block_hash
        apply_sha256(local_block_content, d_strlen((const char*)local_block_content), local_block_hash, 1);

        // compare hashes
        if (compare_hashes(local_block_hash, d_difficulty_5_zeros) <= 0) {
                // if global nonce is 0, then it will take the value of the index, so global nonce will be updated
                // only once
                atomicCAS((unsigned long long*)&global_nonce, (unsigned long long)0, (unsigned long long)index);

                // if global nonce is not 0 anymore, than update the hash value
                if (index == global_nonce) {
                        d_strcpy((char *)block_hash, (const char *)local_block_hash);
                }
        }
}

int main(int argc, char **argv) {
        BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
                        tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
                        tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE];
        BYTE block_hash[SHA256_HASH_SIZE] = "0000000000000000000000000000000000000000000000000000000000000000";
        uint64_t nonce = 0;
        size_t current_length;

        // Top hash
        apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
        apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
        apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
        apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);
        strcpy((char *)tx12, (const char *)hashed_tx1);
        strcat((char *)tx12, (const char *)hashed_tx2);
        apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
        strcpy((char *)tx34, (const char *)hashed_tx3);
        strcat((char *)tx34, (const char *)hashed_tx4);
        apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
        strcpy((char *)tx1234, (const char *)hashed_tx12);
        strcat((char *)tx1234, (const char *)hashed_tx34);
        apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

        // prev_block_hash + top_hash
        strcpy((char*)block_content, (const char*)prev_block_hash);
        strcat((char*)block_content, (const char*)top_hash);
        current_length = strlen((char*) block_content);

        // START IMPLEMENTATION

        // declare parameters for the kernel and allocate Unified Memory for them
        BYTE *d_block_content, *d_block_hash;
        cudaMallocManaged(&d_block_content, BLOCK_SIZE);
        cudaMallocManaged(&d_block_hash, SHA256_HASH_SIZE);

        // initialize d_block_content and d_block_hash
        strcpy((char*)d_block_content, (const char*)block_content);
        strcpy((char*)d_block_hash, (const char*)block_hash);

        //  set block size and the number of blocks so that
        // (block_size * num_blocks >= MAX_NONCE)
        int block_size = 256;
        int num_blocks = (MAX_NONCE + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        startTiming(&start, &stop);

        findNonce<<<num_blocks, block_size>>>(d_block_content, current_length, d_block_hash);

        // wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // copy memory from global memory on gpu to nonce variable on cpu
        cudaMemcpyFromSymbol(&nonce, global_nonce, sizeof(uint64_t));

        float seconds = stopTiming(&start, &stop);
        printResult(d_block_hash, nonce, seconds);

        // free memory
        cudaFree(d_block_content);
        cudaFree(d_block_hash);

        return 0;
}
