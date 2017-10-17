/******************************************************************************
 *   File: main.cpp
 *   Description: Image convolution using OpenCL computing kernel.
 *   Created: 10 oct 2017
 *   Copyright: (C) 2017 Edward Zhornovy <ed@zhornovy.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 
 ******************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#define COMPUTE_KERNEL_FILENAME         ("/Users/user/Documents/opencl/convolution/oclFilter.cl") // full path to file

using namespace std;

const int width = 640,height = 480;       // my webcam resolution, change it to yours one

int err;                            // error output
size_t global;                      // global domain size for our calculation
size_t local;                       // local domain size for our calculation

cl_device_id device_id;             // compute device id
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel kernel;                   // compute kernel
cl_mem input;                       // device memory used for the input array
cl_mem output;                      // device memory used for the output array

unsigned int count2 = 307200;       // 1D array of image data 640*480
int gpu = 1;                        // GPU flag, set 0 for OpenCL computing at your CPU
double avgTime = 0;                 // statistics info | average time for a rendering one image
int counts = 0;                     // sotal count of made computings


static int LoadTextFromFile(const char *file_name, char **result_string, size_t *string_len)
{
    char cCurrentPath[1024];
    getcwd(cCurrentPath, sizeof(cCurrentPath));
    cout << cCurrentPath;
    int fd;
    unsigned file_len;
    struct stat file_status;
    int ret;
    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1){
        printf("Error opening file %s\n", file_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret){
        printf("Error reading status for file %s\n", file_name);
        return -1;
    }
    file_len = (unsigned)file_status.st_size;
    *result_string = (char*)calloc(file_len + 1, sizeof(char));
    ret = (int)read(fd, *result_string, file_len);
    if (!ret){
        printf("Error reading from file %s\n", file_name);
        return -1;
    }
    close(fd);
    *string_len = file_len;
    return 0;
}

int initMyFilterCl()
{
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    char *source = 0;
    size_t length = 0;
    printf("Loading kernel source from file '%s'...\n", "oclFilter.cl");
    err = LoadTextFromFile(COMPUTE_KERNEL_FILENAME, &source, &length);
    if (!source || err)
    {
        printf("Error: Failed to load kernel source!\n");
        return EXIT_FAILURE;
    }
    program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    err = clBuildProgram(program, 0, NULL, "-cl-std=CL1.2", NULL, NULL); // you can use "-cl-std=CL2.0" or "-cl-std=CL1.1" but local workgroups must be equal in v1.1
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[3072000];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    kernel = clCreateKernel(program, "myFilter", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(uchar) * count2, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * count2, NULL, NULL);
    // device memory used for the output array
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    // Detect if your gpu supports double precision
    //
    cl_device_fp_config cfg;
    clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
    printf("\nDouble FP = %llu\n", cfg);
    
    return 0;
}

int computeMyFilterCl(uchar* inputData,uchar* data2)
{
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(uchar) * count2, inputData, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    // Set the arguments to our compute kernel
    //
    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    //cout << "Local Size = " << CL_KERNEL_WORK_GROUP_SIZE<< endl;
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    global = count2;
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL); // NULL  or &local // better NULL
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(uchar) * count2, data2, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    return 0;
}

void releaseMyFilterCl(){
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

void show(const std::string& name, const cv::Mat& mat);
void showMatrix(uchar* data, int width);
void myFilter(uchar* data, uchar* data2);

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "cam open fail" << std::endl;
        return -1;
    }
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 70);
    
    cv::Mat frame, frameGray;
    uchar* data;
    uchar data2[640*480];
    initMyFilterCl();
    for (;;)
    {
        std::clock_t start1;
        double duration1;
        start1 = std::clock();
        cap >> frame;
        cv::cvtColor(frame, frameGray, CV_RGB2GRAY);
        data = frameGray.data;
        showMatrix(data,frameGray.cols);
        computeMyFilterCl(data,data2);
        //myFilter(data,data2);
        showMatrix(data2,frameGray.cols);
        
        duration1 = ( std::clock() - start1 ) / (double) CLOCKS_PER_SEC;
        avgTime+=duration1;
        counts++;
        std::cout << "time: " << duration1 << " avgTime: " << avgTime/counts <<'\n';
        cv::Mat frameOutBefore(frameGray.rows, frameGray.cols, CV_8UC1, data);
        show("frameOutBefore", frameOutBefore);
        cv::Mat frameOut3(frameGray.rows, frameGray.cols, CV_8UC1, data2);
        show("frameOut", frameOut3);

        cv::waitKey(10);
    }
    //releaseMyFilterCl();
    return 0;
}

void showMatrix(uchar* data,int width){
    int sampleI = 300;
    int sampleJ = 120;
    cout << "\n";
    cout << (int)data[simpleI     + sampleJ     * width] << "|" 
         << (int)data[simpleI     + (sampleJ+1) * width] << "|" 
         << (int)data[simpleI     + (sampleJ+2) * width] << endl;
    cout << (int)data[(simpleI+1) + sampleJ     * width] << "|" 
         << (int)data[(simpleI+1) + (sampleJ+1) * width] << "|" 
         << (int)data[(simpleI+1) + (sampleJ+2) * width] << endl;
    cout << (int)data[(simpleI+2) + sampleJ     * width] << "|" 
         << (int)data[(simpleI+2) + (sampleJ+1) * width] << "|" 
         << (int)data[(simpleI+2) + (sampleJ+2) * width] << endl;
}

void show(const std::string& name, const cv::Mat& mat)
{
    if (!mat.empty())
    {
        cv::imshow(name, mat);
    }
}

// One thread cpu function
//
void myFilter(uchar* data,uchar* data2)
{
    static float kernelMatrix[] = {
        -1, -0, 1,
        -2, -0, 2,
        -1, -0, 1};
    
    int kernelWidth = 3;
    int kernelHeight = 3;
    
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            if (x > 20 && x < width-20 && y > 20 && y < height-20 ) {
                double rSum = 0, kSum = 0;
                
                for (int i = 0; i < kernelWidth; i++)
                {
                    for (int j = 0; j < kernelHeight; j++)
                    {  
                        int pixelPosX = x + (i - (kernelWidth / 2));
                        int pixelPosY = y + (j - (kernelHeight / 2));
                        
                        if ((pixelPosX < 0) ||
                            (pixelPosX >= width) ||
                            (pixelPosY < 0) ||
                            (pixelPosY >= height)) continue;
                        
                        auto r = data[pixelPosX + pixelPosY * width];
                        
                        double kernelVal = kernelMatrix[i + j * kernelWidth];
                        rSum += r * kernelVal;
                        
                        kSum += kernelVal;
                    }
                }        
                if (kSum == 0) kSum = 1;
                rSum /= kSum;
                auto rx = (char)rSum;
                data2[x+y*width] = rx;
            }else{
                data2[x+y*width] = data[x+y*width];
            }
        }
    }
}
