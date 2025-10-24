__kernel void EmbeddingLookup(
    __global const int* tokenIds,
    __global const float* embeddingMatrix,
    __global float* output,
    const int dModel)
{
    int i = get_global_id(0); // index in sequence
    int tokenId = tokenIds[i];

    // Each thread copies one embedding row
    for (int j = 0; j < dModel; j++)
    {
        output[i * dModel + j] = embeddingMatrix[tokenId * dModel + j];
    }
}