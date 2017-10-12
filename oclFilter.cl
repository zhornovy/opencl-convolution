__kernel void myFilter(
__global uchar* data, __global uchar* data2){
    float kernelMatrix[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};
    uint kernelWidth = 3;
    uint kernelHeight = 3,width = 640,height = 480;
    uint x = get_global_id(0);
    int y = 0;
    while(x > width){
        x-=width;
        y++;
    }
    if (x>5 && x < width-5 && y>5 && y < height-5 ) {
        float rSum = 0, kSum = 0;
        for (uint i = 0; i < kernelWidth; i++)
        {
            for (uint j = 0; j < kernelHeight; j++)
            {
                int pixelPosX = x + (i - (kernelWidth / 2));
                int pixelPosY = y + (j - (kernelHeight / 2));
                
                if ((pixelPosX < 0) ||
                    (pixelPosX >= width) ||
                    (pixelPosY < 0) ||
                    (pixelPosY >= height)) continue;
                
                uchar r = data[pixelPosX + pixelPosY * width];
                float kernelVal = kernelMatrix[i + j * kernelWidth];
                rSum += r * kernelVal;
                
                kSum += kernelVal;
            }
        }
        if (kSum == 0) kSum = 1;
        rSum /= kSum;
        data2[x+y*width] = (uchar)rSum;
    }else{
        data2[x+y*width] = (uchar)(250);
    }
 barrier(CLK_LOCAL_MEM_FENCE);
}
