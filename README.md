# OpenCL-convolution

Image convolution with a custom matrix using OpenCL computing and GPU utilization.

- [x] multiplatform execution
- [x] maximum GPU utilization
- [x] webcam image usage

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

The initial example requires OpenCV library to capture a raw image that will be used as an input source for a convolution. If you do not have OpenCV you can use any other image with one color channel.
Also, you have to and OpenCL framework to your liker arguments if it is required. 
> Do not forget to install GPU drivers to be able to use OpenCL. Sometimes, generic drivers do not support OpenCL.

Change image resolution if you need it.

```c
int width = 640,height = 480;      // change it to yours one
```

If you do not want to use OpenCV you need to delete loop scopes and replace following code with data array of your image.
```c
 cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "cam open fail" << std::endl;
        return -1;
    }
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 70);
    
    cv::Mat frame, frameGray;
 cap >> frame;
        cv::cvtColor(frame, frameGray, CV_RGB2GRAY);
        data = frameGray.data;
```

## Deployment

Any C++ compiler with C++11 standard or higher.

## Built With

* [OpenCL](https://www.khronos.org/opencl/) - The cross-platform framework using for a parallel computing
* [OpenCV](https://opencv.org/) -  The library mainly aimed at real-time computer vision.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE) file for details.