__kernel void RotaryEmbedding(
    __global float* tensor,    // [rows×dHead], rows = numHeads*seqLen
    const int rows,
    const int seqLen,
    const int dHead)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int pos = row % seqLen;

    for (int i = 0; i < dHead; i += 2)
    {
        float angle = pos / pow(10000.0f, (2.0f * i) / (float)dHead);
        float c = cos(angle);
        float s = sin(angle);

        int idx_even = row * dHead + i;
        int idx_odd  = idx_even + 1;

        float x_even = tensor[idx_even];
        float x_odd  = (i + 1 < dHead) ? tensor[idx_odd] : 0.0f;

        tensor[idx_even] = x_even * c - x_odd * s;
        if (i + 1 < dHead)
            tensor[idx_odd] = x_even * s + x_odd * c;
    }
}
