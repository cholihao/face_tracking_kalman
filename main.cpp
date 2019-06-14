//// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief The entry point for the Inference Engine interactive_face_detection demo application
 * \file interactive_face_detection_demo/main.cpp
 * \example interactive_face_detection_demo/main.cpp
 */
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <thread>
#include <map>
#include <ctime>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "interactive_face_detection.hpp"
#include "detectors.hpp"

#include <ie_iextension.h>
#include <ext_list.hpp>

#include "RedisClient.hpp"
#include <iostream>
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include "H264LiveServerMediaSession.h"
#include "x264Encoder.h"
#include "sort.hpp"
#include "common.hpp"
#include "qfifo.hpp"
#include <curl/curl.h>
#include <sys/stat.h>

using namespace InferenceEngine;

using namespace std;
using namespace cv;
using namespace nlohmann;

//#define DEBUG
ThreadSafeFIFOBuffer<MatData> imagebuf;
//ThreadSafeFIFOBuffer<MatData> imagebuf2;

//Compress video
VideoWriter outputVideo; // For writing the video
VideoWriter outputVideo_depth;
//110219 Defines variables that will be read from the JSON file
int aSize;
int qSize;
int fps;
int no_skip_frames;
int operation_mode;
int set_width;
int set_height;
#define NANN 999999

VerySimpleThreadSafeFIFOBuffer<SendData> sendBuffer(qSize);
//VerySimpleThreadSafeFIFOBuffer<SendData> sendBuffer_depth(qSize);
//010419 Instead of the above buffer size, I will like to increase the depth buffer size so that it does not cause memory leak
VerySimpleThreadSafeFIFOBuffer<SendData> sendBuffer_depth(qSize);

rs2_stream stream_to_align_with;


char clip_fname[80];
char clip_fname_depth[80];
char jpg_fname[80];
FILE *clip;

int cnt=0;
int event_cnt=NANN;
int frame_no=0;
int concatenate_flag = 0; //Keeps tracks if we should concatenate the files or not
int previous_max; //Keep tracks of overlapped frames
int previous_cnt;

inline bool file_exist(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}


// need to seek where is the IDR frame otherwise which will be the broken video clip
//
void *save_one_frame(void *frame)
{
    char fname[80];
    SendData *framedata = (SendData*)frame;
    pthread_detach(pthread_self());
    vector<int> params;
    const int JPEG_QUALITY = 80;
    //params.push_back(cv::CV_IMWRITE_JPEG_QUALITY);
    params.push_back(JPEG_QUALITY);
    if ((file_exist("/tmp/event") || concatenate_flag>0 ) && operation_mode == 0) {
        if (file_exist("/tmp/event")) {
            if (concatenate_flag == 0) {
                //Initialize the H264 compressed video
                sprintf(clip_fname, "event_x%06d_s%04d_e%04d.264",cnt,framedata->vars.idx,cnt+aSize-1);
                previous_max = cnt+aSize-1;
                int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
                //int fps = 6;
                Size S = Size(framedata->vars.cols, framedata->vars.rows);
                outputVideo.open(clip_fname, fourcc, fps, S);
                unlink("/tmp/event");
                cout << "\033[92m"<< clip_fname << "\033[90m" << endl;
                concatenate_flag = 1;
                //*event_cnt = aSize+qSize-1;
                event_cnt=aSize+qSize;
            }
            else {
                cout << "Concatenate with previous video " << clip_fname << endl;
                printf("Second event ID: event_x%06d_s%04d_e%04d.264 \n",cnt,framedata->vars.idx,cnt+aSize-1);
                int current_min = cnt-qSize;
                //*int overlapped_frames = previous_max - current_min;
                int overlapped_frames = previous_max - current_min + 1 ;
                unlink("/tmp/event");
                concatenate_flag += 1;
                //*event_cnt = event_cnt + aSize+qSize-1 - overlapped_frames; //Takes care of the overlapped frames
                event_cnt = event_cnt+aSize+qSize-overlapped_frames; //Takes care of the overlapped frames
                cout << "No. of videos combined: " << concatenate_flag << endl;
                previous_max = cnt + aSize-1;
            }
        }

        event_cnt--;
        Mat received_frame(Size(framedata->vars.cols, framedata->vars.rows), CV_8UC3, framedata->vars.buf1, Mat::AUTO_STEP);
        cout << "Event Counter: " << event_cnt << endl;
        outputVideo.write(received_frame);

        if (event_cnt==0) {
            concatenate_flag = 0; //Has to put this before outputVideo.release() function, this is because of parallel threads
            outputVideo.release();
            cout << "waiting for another event" << endl;
        }

        free (framedata->vars.buf1);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //080219 If operation_mode == 1, following code skips the frames as long as the second event is within no_skip_frames
    else if ((file_exist("/tmp/event") || concatenate_flag>0 ) && operation_mode == 1) {
        if (file_exist("/tmp/event")) {
            if (concatenate_flag == 0) {
                //Initialize the H264 compressed video
                sprintf(clip_fname, "event_x%06d_s%04d_e%04d.264",cnt,framedata->vars.idx,cnt+aSize-1);
                previous_max = cnt+aSize-1;
                previous_cnt = cnt;
                int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
                //int fps = 6;
                Size S = Size(framedata->vars.cols, framedata->vars.rows);
                outputVideo.open(clip_fname, fourcc, fps, S);
                unlink("/tmp/event");
                cout << "\033[92m"<< clip_fname << "\033[90m" << endl;
                concatenate_flag = 1;
                //*event_cnt = aSize+qSize-1;
                event_cnt=aSize+qSize;
            }
            else if ((cnt - previous_cnt) >  no_skip_frames) {

                cout << "Concatenate with previous video as the current cnt is more than the no of skip frames " << clip_fname << endl;
                printf("Second event ID: event_x%06d_s%04d_e%04d.264 \n",cnt,framedata->vars.idx,cnt+aSize-1);
                int current_min = cnt-qSize;
                //*int overlapped_frames = previous_max - current_min;
                int overlapped_frames = previous_max - current_min + 1;
                unlink("/tmp/event");
                concatenate_flag += 1;
                //*event_cnt = event_cnt + aSize+qSize-1 - overlapped_frames; //Takes care of the overlapped frames
                event_cnt = event_cnt+aSize+qSize-overlapped_frames;
                cout << "No. of videos combined: " << concatenate_flag << endl;
                previous_max = cnt + aSize-1;
                previous_cnt = cnt;
            }
            else {
                cout << "Does not concatenate with previous video as the current cnt is lesser than the no of skip frames " << endl;
                unlink("/tmp/event");
            }
        }
        event_cnt--;
        Mat received_frame(Size(framedata->vars.cols, framedata->vars.rows), CV_8UC3, framedata->vars.buf1, Mat::AUTO_STEP);
        cout << "Event Counter: " << event_cnt << endl;
        outputVideo.write(received_frame);

        if (event_cnt==0) {
            concatenate_flag = 0; //Has to put this before outputVideo.release() function, this is because of parallel threads
            outputVideo.release();
            cout << "waiting for another event" << endl;
        }

        free (framedata->vars.buf1);


    } else {
        event_cnt=NANN;
        concatenate_flag = 0;
        free (framedata->vars.buf1);
    }
    delete framedata;
    return 0;
}




void *save_frame(void *)
{
    int error;
    int i = 0;
    pthread_t tid;
    SendData sendData;

    while (1)
    {
        SendData *sendData = new SendData;

        cout << "Size of the color buffer is: " << sendBuffer.Size() << endl;

        if (sendBuffer.Size() > qSize-1) {
            //cout << "buffer Size " << sendBuffer.Size() << endl;
            if(!sendBuffer.Pop(*sendData))
            {
                delete sendData;
                usleep(10);
                continue;
            }

            //010419 Masked the following block for debugging

            pthread_t tid;
            error = pthread_create(&tid, NULL, save_one_frame, (void*)sendData);
            if(0 != error)
                cerr << "Couldn't run thread number " << tid << " , errno " <<  error << endl;

            //010419 Free the buffer since there is no saving action
            //free (sendData->vars.buf1);
            //free (sendData_depth->vars.buf1);
        } else {
            usleep(10000L);
            //010419 Solving the memory leak issue
            delete sendData;
        }

        //010419 Tries to solve the memory leak issue
        //delete sendData;

    }
    cout << "Flushing send buffer......" << endl;
    while (sendBuffer.Pop(sendData))
    {
        cout << ".";
    }
    cout << "Flushing: send done!" << endl;

    //310319 Added the following, hope that it would solve the memory loss issue
    //free (sendData.vars.buf1);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Save both the color and depth frames
void *save_one_color_depth_frame(void *frame)
{
    struct sendData_color_depth *testing = (struct sendData_color_depth *)frame;
    SendData *framedata = testing->rs_color_frame;
    SendData *framedata_depth = testing->rs_depth_frame;
    char fname[80];
    char fname_depth[80];
    //SendData *framedata = (SendData*)frame;
    pthread_detach(pthread_self());
    vector<int> params;
    const int JPEG_QUALITY = 80;
    //params.push_back(cv::CV_IMWRITE_JPEG_QUALITY);
    params.push_back(JPEG_QUALITY);
    if ((file_exist("/tmp/event") || concatenate_flag>0 ) && operation_mode == 0) {
        if (file_exist("/tmp/event")) {
            if (concatenate_flag == 0) {
                //Initialize the H264 compressed video
                sprintf(clip_fname, "event_x%06d_s%04d_e%04d.264",cnt,framedata->vars.idx,cnt+aSize-1);
                sprintf(clip_fname_depth, "event_depth_x%06d_s%04d_e%04d.264",cnt,framedata_depth->vars.idx+3,cnt+aSize-1);
                previous_max = cnt+aSize-1;
                int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
                //int fps = 6;
                Size S = Size(framedata->vars.cols, framedata->vars.rows);
                Size S_depth = Size(framedata_depth->vars.cols, framedata_depth->vars.rows);
                outputVideo.open(clip_fname, fourcc, fps, S);
                outputVideo_depth.open(clip_fname_depth, fourcc, fps, S_depth);
                unlink("/tmp/event");
                cout << "\033[92m"<< clip_fname << "\033[90m" << endl;
                cout << "\033[92m"<< clip_fname_depth << "\033[90m" << endl;
                concatenate_flag = 1;
                //*event_cnt = aSize+qSize-1;
                event_cnt=aSize+qSize;
            }
            else {
                cout << "Concatenate with previous video " << clip_fname << endl;
                printf("Second event ID: event_x%06d_s%04d_e%04d.264 \n",cnt,framedata->vars.idx,cnt+aSize-1);
                printf("Second event ID: event_depth_x%06d_s%04d_e%04d.264 \n",cnt,framedata_depth->vars.idx+3,cnt+aSize-1);
                int current_min = cnt-qSize;
                //*int overlapped_frames = previous_max - current_min;
                int overlapped_frames = previous_max - current_min + 1 ;
                unlink("/tmp/event");
                concatenate_flag += 1;
                //*event_cnt = event_cnt + aSize+qSize-1 - overlapped_frames; //Takes care of the overlapped frames
                event_cnt = event_cnt+aSize+qSize-overlapped_frames; //Takes care of the overlapped frames
                cout << "No. of videos combined: " << concatenate_flag << endl;
                previous_max = cnt + aSize-1;
            }
        }
        event_cnt--;
        Mat received_frame(Size(framedata->vars.cols, framedata->vars.rows), CV_8UC3, framedata->vars.buf1, Mat::AUTO_STEP);
        Mat received_frame_depth(Size(framedata_depth->vars.cols, framedata_depth->vars.rows), CV_8UC3, framedata_depth->vars.buf1, Mat::AUTO_STEP);
        cout << "Event Counter: " << event_cnt << endl;
        //Save as jpg frame for verification
        /*
           sprintf(jpg_fname, "event_x%06d_s%04d_e%04d.jpg",cnt,framedata->vars.idx,cnt+aSize-1);
           imwrite(jpg_fname, received_frame, params);
           */
        outputVideo.write(received_frame);
        outputVideo_depth.write(received_frame_depth);

        if (event_cnt==0) {
            concatenate_flag = 0; //Has to put this before outputVideo.release() function, this is because of parallel threads
            outputVideo.release();
            outputVideo_depth.release();
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //190219 Required for sending of videos to the server
            CURL *curl;
            CURLcode res;
            struct curl_httppost *formpost=NULL;
            struct curl_httppost *lastptr=NULL;
            //struct curl_slist *headerlist=NULL;
            long respcode; //response code of the http transaction
            //static const char buf[] = "Expect:";

            CURLFORMcode code;
            CURLFORMcode code_depth;
            curl_global_init(CURL_GLOBAL_ALL);
            code = curl_formadd(&formpost,&lastptr,
                    CURLFORM_COPYNAME, "scene-event",
                    //CURLFORM_FILE, "event_0042_s0013_e0071.264",
                    CURLFORM_FILE, clip_fname,
                    CURLFORM_END);

            code_depth = curl_formadd(&formpost,&lastptr,
                    CURLFORM_COPYNAME, "depth-event",
                    //CURLFORM_FILE, "event_depth_0042_s0013_e0071.264",
                    CURLFORM_FILE, clip_fname_depth,
                    CURLFORM_END);
            if(code != 0 || code_depth !=0){
                printf("Something went wrong in formadd.\n");
            }
            else
                cout << "code is: " << code << " code_depth is: " << code_depth << endl;
            curl = curl_easy_init();
            cout << "curl return code is: " << curl << endl;
            //headerlist = curl_slist_append(headerlist, buf);
            if(curl) {

                auto srv_ip = getenv("SRV_IP");
                if (!srv_ip) {
                    cout << "Please export SRV_IP=http://IP:8000/va/pdpa/video" << endl;
                    return 0;
                }
                curl_easy_setopt(curl, CURLOPT_URL, srv_ip);
                curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
                res = curl_easy_perform(curl);
                if(res != CURLE_OK)
                {
                    cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
                }
                curl_easy_getinfo(curl,CURLINFO_RESPONSE_CODE, &respcode);// grabbing it from curl
                if(respcode == 200)
                {
                    cout << "*" <<endl;
                }

                curl_easy_cleanup(curl);
                curl_formfree(formpost);
                //curl_slist_free_all (headerlist);
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            cout << "waiting for another event" << endl;
        }

        //310319 If you are using copyTo function, there is no allocated memory using malloc, thus you have to masked these two lines off
        free (framedata->vars.buf1);
        free (framedata_depth->vars.buf1);
    }
    else if ((file_exist("/tmp/event") || concatenate_flag>0 ) && operation_mode == 1) {
        if (file_exist("/tmp/event")) {
            if (concatenate_flag == 0) {
                //Initialize the H264 compressed video
                sprintf(clip_fname, "event_x%06d_s%04d_e%04d.264",cnt,framedata->vars.idx,cnt+aSize-1);
                sprintf(clip_fname_depth, "event_depth_x%06d_s%04d_e%04d.264",cnt,framedata_depth->vars.idx+3,cnt+aSize-1);
                previous_max = cnt+aSize-1;
                previous_cnt = cnt;
                int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
                //int fps = 6;
                Size S = Size(framedata->vars.cols, framedata->vars.rows);
                Size S_depth = Size(framedata_depth->vars.cols, framedata_depth->vars.rows);
                outputVideo.open(clip_fname, fourcc, fps, S);
                outputVideo_depth.open(clip_fname_depth, fourcc, fps, S_depth);
                unlink("/tmp/event");
                cout << "\033[92m"<< clip_fname << "\033[90m" << endl;
                cout << "\033[92m"<< clip_fname_depth << "\033[90m" << endl;
                concatenate_flag = 1;
                //*event_cnt = aSize+qSize-1;
                event_cnt=aSize+qSize;
            }
            else if ((cnt - previous_cnt) >  no_skip_frames) {

                cout << "Concatenate with previous video as the current cnt is more than the no of skip frames " << clip_fname << endl;
                printf("Second event ID: event_x%06d_s%04d_e%04d.264 \n",cnt,framedata->vars.idx,cnt+aSize-1);
                printf("Second event ID: event_depth_x%06d_s%04d_e%04d.264 \n",cnt,framedata_depth->vars.idx+3,cnt+aSize-1);
                int current_min = cnt-qSize;
                //*int overlapped_frames = previous_max - current_min;
                int overlapped_frames = previous_max - current_min + 1;
                unlink("/tmp/event");
                concatenate_flag += 1;
                //*event_cnt = event_cnt + aSize+qSize-1 - overlapped_frames; //Takes care of the overlapped frames
                event_cnt = event_cnt+aSize+qSize-overlapped_frames;
                cout << "No. of videos combined: " << concatenate_flag << endl;
                previous_max = cnt + aSize-1;
                previous_cnt = cnt;
            }
            else {
                cout << "Does not concatenate with previous video as the current cnt is lesser than the no of skip frames " << endl;
                unlink("/tmp/event");
            }
        }
        event_cnt--;
        Mat received_frame(Size(framedata->vars.cols, framedata->vars.rows), CV_8UC3, framedata->vars.buf1, Mat::AUTO_STEP);
        Mat received_frame_depth(Size(framedata_depth->vars.cols, framedata_depth->vars.rows), CV_8UC3, framedata_depth->vars.buf1, Mat::AUTO_STEP);
        cout << "Event Counter: " << event_cnt << endl;
        //Save as jpg frame for verification
        /*
           sprintf(jpg_fname, "event_x%06d_s%04d_e%04d.jpg",cnt,framedata->vars.idx,cnt+aSize-1);
           imwrite(jpg_fname, received_frame, params);
           */
        outputVideo.write(received_frame);
        outputVideo_depth.write(received_frame_depth);

        if (event_cnt==0) {
            concatenate_flag = 0; //Has to put this before outputVideo.release() function, this is because of parallel threads
            outputVideo.release();
            outputVideo_depth.release();
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //190219 Required for sending of videos to the server
            CURL *curl;
            CURLcode res;
            struct curl_httppost *formpost=NULL;
            struct curl_httppost *lastptr=NULL;
            //struct curl_slist *headerlist=NULL;
            long respcode; //response code of the http transaction
            //static const char buf[] = "Expect:";

            CURLFORMcode code;
            CURLFORMcode code_depth;
            curl_global_init(CURL_GLOBAL_ALL);
            code = curl_formadd(&formpost,&lastptr,
                    CURLFORM_COPYNAME, "scene-event",
                    //CURLFORM_FILE, "event_0042_s0013_e0071.264",
                    CURLFORM_FILE, clip_fname,
                    CURLFORM_END);

            code_depth = curl_formadd(&formpost,&lastptr,
                    CURLFORM_COPYNAME, "depth-event",
                    //CURLFORM_FILE, "event_depth_0042_s0013_e0071.264",
                    CURLFORM_FILE, clip_fname_depth,
                    CURLFORM_END);

            if(code != 0 || code_depth !=0){
                printf("Something went wrong in formadd.\n");
            }
            else
                cout << "code is: " << code << " code_depth is: " << code_depth << endl;
            curl = curl_easy_init();
            cout << "curl return code is: " << curl << endl;
            //headerlist = curl_slist_append(headerlist, buf);
            if(curl) {

                auto srv_ip = getenv("SRV_IP");
                if (!srv_ip) {
                    cout << "Please export SRV_IP=http://IP:8000/va/pdpa/video" << endl;
                    return 0;
                }
                curl_easy_setopt(curl, CURLOPT_URL, srv_ip);
                curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
                res = curl_easy_perform(curl);
                if(res != CURLE_OK)
                {
                    cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
                }
                curl_easy_getinfo(curl,CURLINFO_RESPONSE_CODE, &respcode);// grabbing it from curl

                if(respcode == 200)
                {
                    cout << "*" <<endl;
                }

                curl_easy_cleanup(curl);
                curl_formfree(formpost);
            }
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            cout << "waiting for another event" << endl;
        }

        //310319 If you are using copyTo function, there is no allocated memory using malloc, thus you have to masked these two lines off
        free (framedata->vars.buf1);
        free (framedata_depth->vars.buf1);


    } else {
        event_cnt=NANN;
        concatenate_flag = 0;

        //020419 Tries to solve the memory leak issue which is not reflected in valgrind
        free (framedata->vars.buf1);
        free (framedata_depth->vars.buf1);

    }

    //010419 I masked the following to check if there are double deletion

    delete testing;
    delete framedata;
    delete framedata_depth;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//310319 Modified, such that you only run this when there is no DEBUG
#ifndef DEBUG
//130219 Function which pops out BOTH color and depth data
void *save_frame_color_depth(void *)
{
    int error;
    int i = 0;

    pthread_t tid;
    SendData sendData;
    SendData sendData_depth;

    while (1)
    {
        sendData_color_depth *color_depth_struct = new sendData_color_depth;
        color_depth_struct->rs_color_frame = new SendData;
        color_depth_struct->rs_depth_frame = new SendData;

        //310319 I will like to check the buffer size
        //cout << "Size of the color buffer is: " << sendBuffer.Size() << endl;
        //cout << "Size of the depth buffer is: " << sendBuffer_depth.Size() << endl;

        if (sendBuffer.Size() > qSize-1 && sendBuffer_depth.Size() > qSize-1) {
            if(!sendBuffer.Pop(*color_depth_struct->rs_color_frame) || !sendBuffer_depth.Pop(*color_depth_struct->rs_depth_frame))
            {
                delete color_depth_struct->rs_color_frame;
                delete color_depth_struct->rs_depth_frame;
                //010419 Delete the color_depth_struct pointer
                delete color_depth_struct;
                usleep(10);
                continue;
            }

            //010419 Locating the memory leak issue

            pthread_t tid;
            error = pthread_create(&tid, NULL, save_one_color_depth_frame, static_cast<sendData_color_depth *>(color_depth_struct));
            if(0 != error)
                cerr << "Couldn't run thread number " << tid << " , errno " <<  error << endl;


        } else {
            usleep(10000L);
                //010419 I delete the pointers here when the buffer size is not full, hopefully this is the reason for the memory leak issue
                delete color_depth_struct->rs_color_frame;
                delete color_depth_struct->rs_depth_frame;
                //010419 Delete the color_depth_struct pointer
                delete color_depth_struct;

        }

                //010419 Delete all the pointers that have been created
                //010419 Masked the following block
                /*
                delete color_depth_struct->rs_color_frame;
                delete color_depth_struct->rs_depth_frame;
                delete color_depth_struct;
                */


    }
    cout << "Flushing send buffer......" << endl;

    while (sendBuffer.Pop(sendData) || sendBuffer_depth.Pop(sendData_depth))
    {
        cout << ".";
    }

    cout << "Flushing: send done!" << endl;
    return 0;
}
//310319 Only run the above function when there is no DEBUG
#endif

static bool doMosaic(Mat &img, int msize = 5)
{
    for (int i = 0; i < img.cols-msize; i+=msize) {
        for(int j = 0; j < img.rows-msize; j+=msize) {
            Rect r = Rect(i,j,msize,msize);
            Mat mosaic = img(r);
            mosaic.setTo(mean(mosaic));
        }
    }
    return true;
}

static inline bool read_image(rs2::pipeline pipe, rs2::colorizer color_map, cv::Mat &frame) {

    rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    //rs2::frame color_frame = data.get_color_frame();
    rs2::align align(stream_to_align_with);
    rs2::frameset processed = align.process(data);
    rs2::frame color_frame = processed.first(stream_to_align_with);
    rs2::frame depth_frame = processed.get_depth_frame();

    int w = color_frame.as<rs2::video_frame>().get_width();
    int h = color_frame.as<rs2::video_frame>().get_height();
    int stride = color_frame.as<rs2::video_frame>().get_stride_in_bytes();

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);

    Mat image(Size(w, h), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
    //putText(image, "Frame No: ", Point(50,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(252,15,192), 4);
    //putText(image, std::to_string(frame_no), Point(230,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(252,15,192), 4);
    //rs2::frame depth_frame = data.get_depth_frame(); // Find the depth data
    depth_frame = color_map.process(depth_frame);

    Mat depth_image(Size(w, h), CV_8UC3, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
    Mat depth_BGR;
    cvtColor(depth_image, depth_BGR, cv::COLOR_RGB2BGR); //Has to convert as colorize changes depth to RGB8 and not BGR. See following link for more info:
    //http://docs.ros.org/kinetic/api/librealsense2/html/rs__processing_8h.html#aafbda54891983e595b83eeb5648dbeb5
    putText(depth_BGR, str, Point(230,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(252,15,192), 4);

    Size s = image.size();
    long framesize;
    int rows = s.height;
    int cols = s.width;
    int no_of_channels = 3;

    unsigned char *image_pointer_depth;
    framesize = rows * cols * no_of_channels;
    SendData sendData_depth;



    image_pointer_depth = depth_BGR.data;
    unsigned char *buf1_depth = (unsigned char*) malloc (sizeof (unsigned char) * framesize);
    memcpy(buf1_depth, image_pointer_depth, framesize);
    sendData_depth.vars.buf1 = buf1_depth;



    //cout << "Stored frame size is: " << framesize << endl;
    //cout << "Actual size is: " << depth_BGR.rows << " " << depth_BGR.cols << endl;

    //300319 Use copyTo instead of memcpy instead of the above block
    //Mat bimage;
    //image_pointer_depth = depth_BGR.data;
    //depth_BGR.clone(bimage);
    //bimage = depth_BGR.clone();

    //sendData_depth.vars.buf1 = bimage.data;
    sendData_depth.vars.len = framesize;
    sendData_depth.vars.idx = cnt;
    sendData_depth.vars.cols = cols;
    sendData_depth.vars.rows = rows;

    sendBuffer_depth.Push(sendData_depth);
    //310319 In order to enusre that the depth and color frames are synchronized, I will only want to push the depth data into the buffer after cnt == 0 instead of the above line
    /*
    if (cnt > 0)
    {
         sendBuffer_depth.Push(sendData_depth);
    }
    else
    {
        free (sendData_depth.vars.buf1);
    }
    */




    image.copyTo(frame);
    //frame = image.clone();
    return true;
}

static inline bool read_mosicimage_from_frame(cv::Mat &image) {

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    char fname_in_ram[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);

    cnt++;
    Size s = image.size();
    long framesize;
    int rows = s.height;
    int cols = s.width;
    int no_of_channels = 3;
    unsigned char *image_pointer;
    unsigned char *image_pointer_depth;
    framesize = rows * cols * no_of_channels;
    SendData sendData;

    putText(image, str, Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(252,15,192), 4);

    image_pointer = image.data;
    unsigned char *buf1 = (unsigned char*) malloc (sizeof (unsigned char) * framesize);
    memcpy(buf1, image_pointer, framesize);
    sendData.vars.buf1 = buf1;


    //310319 Instead of the above paragraph, I would changed it to use copyTo function
    //Mat bimage_color;
    //image.clone(bimage_color);
    //bimage_color = image.clone();
    //sendData.vars.buf1 = bimage_color.data;

    sendData.vars.len = framesize;
    sendData.vars.idx = cnt;
    sendData.vars.cols = cols;
    sendData.vars.rows = rows;

    sendBuffer.Push(sendData);
    // save loop image for later use for example ROI
    sprintf(fname_in_ram,"/dev/shm/roi_%02d.png",frame_no % 6);
    cv::imwrite(fname_in_ram,image);

    frame_no += 1;
    usleep(10000L);

    return true;
}

static void RTSPrun() {
    TaskScheduler* taskSchedular = BasicTaskScheduler::createNew();
    BasicUsageEnvironment* usageEnvironment = BasicUsageEnvironment::createNew(*taskSchedular);
    RTSPServer* rtspServer = RTSPServer::createNew(*usageEnvironment, 8554, NULL);
    if(rtspServer == NULL)
    {
        *usageEnvironment << "Failed to create rtsp server ::" << usageEnvironment->getResultMsg() <<"\n";
        exit(1);
    }

    std::string streamName = "pdpa";
    ServerMediaSession* sms = ServerMediaSession::createNew(*usageEnvironment, streamName.c_str(), streamName.c_str(), "PDPDA Live Stream");
    H264LiveServerMediaSession *liveSubSession = H264LiveServerMediaSession::createNew(*usageEnvironment, true);
    sms->addSubsession(liveSubSession);
    rtspServer->addServerMediaSession(sms);
    char* url = rtspServer->rtspURL(sms);
    *usageEnvironment << "Play the stream using url "<<url << "\n";
    delete[] url;
    taskSchedular->doEventLoop();
}


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}

long get_current_time() {
    auto duration  = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    //std::cout << "now " << ms.count() << std::endl;
    return ms.count();
}

int main(int argc, char *argv[]) {

    long framesize;
    char *memblock;
    char fname [80];
    char *rs_frame;
    char *x;
    size_t tail_number;
    string user_interrupts;
    int read_counter;
    int total_frames;
    int frame_no = 0; //Records the frame number
    //void *image_pointer;
    unsigned char *image_pointer;
    unsigned char *image_pointer_depth;
    //const int set_width = 640;
    //const int set_height = 480;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //110219 Reads the variables from the JSON file
    if (!file_exist("parameters.json")) {
        cout<<"parameters.json not founded!!! <Quit>"<<endl;
        return 0;
    }

    std::ifstream in("parameters.json");
    json file = json::parse(in);
    std::cout << std::setw(4) << file << '\n';
    json j;
    j = file;
    aSize = j["aSize"];
    qSize = j["qSize"];
    fps = j["fps"];
    no_skip_frames = j["no_skip_frames"];
    set_height = j["set_height"];
    set_width = j["set_width"];
    operation_mode = j["operation_mode"];

    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        //cv::VideoCapture cap;
        //const bool isCamera = FLAGS_i == "cam";
        //if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
        //    throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        //}
        //

#ifndef DEBUG
        /////////////////////////////////////////////////////////////////////////////////
        //040219 Camera Initialization
        rs2::context ctx;
        auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
        if (list.size() == 0)
            throw std::runtime_error("No device detected. Is it plugged in?");
        rs2::config cfg;
        rs2::pipeline pipe;
        //cfg1.enable_stream(RS2_STREAM_COLOR, -1, 640, 480, RS2_FORMAT_BGR8, 6);
        cfg.enable_stream(RS2_STREAM_COLOR, -1, set_width, set_height, RS2_FORMAT_BGR8, 6);

        cfg.enable_stream(RS2_STREAM_DEPTH, -1, set_width, set_height, RS2_FORMAT_Z16, 6);
        rs2::colorizer color_map;
        //130219 Includes options for the color_map
        color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f);
        color_map.set_option(RS2_OPTION_COLOR_SCHEME, 1.f); //Classic color
        color_map.set_option(RS2_OPTION_VISUAL_PRESET, 0.f); //1.0f is fixed range, if it is 0.f it is dynamic
        color_map.set_option(RS2_OPTION_MIN_DISTANCE, 0.0f);
        color_map.set_option(RS2_OPTION_MAX_DISTANCE, 10.0f);

        rs2::pipeline_profile profile = pipe.start(cfg);
        std::vector<rs2::stream_profile> devicestreams = profile.get_streams();
        bool color_stream_found = false;
        bool depth_stream_found = false;
        //rs2_stream stream_to_align_with;
        for (rs2::stream_profile sp : devicestreams)
        {
            rs2_stream profile_stream = sp.stream_type();
            cout << "Available stream profile is: " << profile_stream << endl;
            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
                stream_to_align_with = RS2_STREAM_COLOR;
            }
            if (profile_stream == RS2_STREAM_DEPTH)
            {
                depth_stream_found = true;
            }
        }
        if (!depth_stream_found || !color_stream_found)
        {
            cout << "color_stream_found is: " << color_stream_found << " depth_stream_found is: " << depth_stream_found << endl;
            throw std::runtime_error("There isn't depth or color stream for alignment");
        }

        const size_t width  = 640;
        const size_t height = 480;
#endif
        // read input (video) frame
        //imagebuf.setdata(1000);
        //std::thread rtsp_thread(RTSPrun);

        cv::Mat frame;

#ifdef DEBUG
        VideoCapture cap("test_858_480.avi");
        size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cap.read(frame);

#else
        if (!read_image(pipe, color_map, frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
#endif


        pthread_t t_send;
        //pthread_create(&t_send, NULL, save_frame, NULL);
        //130219 Instead of the above, I will like to save both color and depth frames
//310319 Added the condition here, if not there will be memory leak in DEBUG mode
#ifdef DEBUG
        //010419 Masked the following so that to check for memory leak
        pthread_create(&t_send, NULL, save_frame, NULL);
#else
        pthread_create(&t_send, NULL, save_frame_color_depth, NULL);
#endif
//310319 End condition

//
        //Disable Video Writer
        //VideoWriter outvideo("outvideo.avi", cv::VideoWriter::fourcc('h','2','6','4'), 10, Size(width, height));
        //VideoWriter outtracking("tracking_result.avi", cv::VideoWriter::fourcc('X','V','I','D'), 20, Size(width, height));
        // ---------------------------------------------------------------------------------------------------
        // --------------------------- 1. Loading plugin to the Inference Engine -----------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp},
            {FLAGS_d_em, FLAGS_m_em}, {FLAGS_d_lm, FLAGS_m_lm}
        };
        FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r);
        AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async);
        EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async);
        FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async);

        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Loading extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Loading extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            pluginsForDevices[deviceName] = plugin;
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // ---------------------------------------------------------------------------------------------------

        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        // Disable dynamic batching for face detector as it processes one image at a time
        Load(faceDetector).into(pluginsForDevices[FLAGS_d], false);
        Load(ageGenderDetector).into(pluginsForDevices[FLAGS_d_ag], FLAGS_dyn_ag);
        Load(headPoseDetector).into(pluginsForDevices[FLAGS_d_hp], FLAGS_dyn_hp);
        Load(emotionsDetector).into(pluginsForDevices[FLAGS_d_em], FLAGS_dyn_em);
        Load(facialLandmarksDetector).into(pluginsForDevices[FLAGS_d_lm], FLAGS_dyn_lm);
        // ----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Doing inference -----------------------------------------------------
        // Starting inference & calculating performance
        slog::info << "Start inference " << slog::endl;
        if (!FLAGS_no_show) {
            std::cout << "Press any key to stop" << std::endl;
        }
        bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
            emotionsDetector.enabled() || facialLandmarksDetector.enabled();

        Timer timer;
        timer.start("total");

        std::ostringstream out;
        size_t framesCounter = 0;
        bool frameReadStatus;
        bool isLastFrame;
        cv::Mat prev_frame, next_frame;
        // Detecting all faces on the first frame and reading the next one
        SORT sorter(15);
        timer.start("detection");
        faceDetector.enqueue(frame);
        faceDetector.submitRequest();
        timer.finish("detection");

        prev_frame = frame.clone();

        // Reading the next frame
        timer.start("video frame decoding");
#ifdef DEBUG
        frameReadStatus = cap.read(frame);
#else
        frameReadStatus = read_image(pipe, color_map, frame);
#endif
        //frameReadStatus = read_image(frame);

        timer.finish("video frame decoding");


        RedisClient *redisClient = new RedisClient("127.0.0.1", 6379);
        redisClient->start();

        while (true) {
            framesCounter++;
            isLastFrame = !frameReadStatus;

            nlohmann::json j;
            bool redis_send = false;
            j["ts"] = get_current_time();
            auto jObjects = nlohmann::json::array();

            timer.start("detection");
            // Retrieving face detection results for the previous frame
            faceDetector.wait();
            faceDetector.fetchResults();
            auto prev_detection_results = faceDetector.results;

            // No valid frame to infer if previous frame is the last
            if (!isLastFrame) {
                faceDetector.enqueue(frame);
                faceDetector.submitRequest();
            }
            timer.finish("detection");

            timer.start("data preprocessing");
            // Filling inputs of face analytics networks
            for (auto &&face : prev_detection_results) {
                if (isFaceAnalyticsEnabled) {
                    auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                    cv::Mat face = prev_frame(clippedRect);
                    ageGenderDetector.enqueue(face);
                    headPoseDetector.enqueue(face);
                    emotionsDetector.enqueue(face);
                    facialLandmarksDetector.enqueue(face);
                }
            }
            timer.finish("data preprocessing");

            // Running Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, and Facial Landmarks Estimation networks simultaneously
            timer.start("face analytics call");
            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.submitRequest();
                headPoseDetector.submitRequest();
                emotionsDetector.submitRequest();
                facialLandmarksDetector.submitRequest();
            }
            timer.finish("face analytics call");

            // Reading the next frame if the current one is not the last
            if (!isLastFrame) {
                timer.start("video frame decoding");
#ifdef DEBUG
                frameReadStatus = cap.read(next_frame);
#else
                frameReadStatus = read_image(pipe, color_map, next_frame);
#endif
                //frameReadStatus = read_image(next_frame);

                timer.finish("video frame decoding");
            }

            timer.start("face analytics wait");
            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.wait();
                headPoseDetector.wait();
                emotionsDetector.wait();
                facialLandmarksDetector.wait();
            }
            timer.finish("face analytics wait");

            // Visualizing results
            if (!FLAGS_no_show) {
                timer.start("visualization");
                /*
                   out.str("");
                   out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                   << (timer["video frame decoding"].getSmoothedDuration() +
                   timer["visualization"].getSmoothedDuration())
                   << " ms";
                   cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                   cv::Scalar(255, 0, 0));

                   out.str("");
                   out << "Face detection time: " << std::fixed << std::setprecision(2)
                   << timer["detection"].getSmoothedDuration()
                   << " ms ("
                   << 1000.f / (timer["detection"].getSmoothedDuration())
                   << " fps)";
                   cv::putText(prev_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                   cv::Scalar(255, 0, 0));
                   */
                if (isFaceAnalyticsEnabled) {
                    /*
                       out.str("");
                       out << "Face Analysics Networks "
                       << "time: " << std::fixed << std::setprecision(2)
                       << timer["face analytics call"].getSmoothedDuration() +
                       timer["face analytics wait"].getSmoothedDuration()
                       << " ms ";
                       if (!prev_detection_results.empty()) {
                       out << "("
                       << 1000.f / (timer["face analytics call"].getSmoothedDuration() +
                       timer["face analytics wait"].getSmoothedDuration())
                       << " fps)";
                       }
                       cv::putText(prev_frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                       cv::Scalar(255, 0, 0));
                       */
                }

                // For every detected face
                int i = 0;
                vector<TrackingBox> detFrameData;
                for (auto &result : prev_detection_results) {
                    cv::Rect rect = result.location;
                    /*
                       out.str("");

                       if (ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch) {
                       out << (ageGenderDetector[i].maleProb > 0.5 ? "M" : "F");
                       out << std::fixed << std::setprecision(0) << "," << ageGenderDetector[i].age;
                       if (FLAGS_r) {
                       std::cout << "Predicted gender, age = " << out.str() << std::endl;
                       }
                       } else {
                       out << (result.label < faceDetector.labels.size() ? faceDetector.labels[result.label] :
                       std::string("label #") + std::to_string(result.label))
                       << ": " << std::fixed << std::setprecision(3) << result.confidence;
                       }

                       if (emotionsDetector.enabled() && i < emotionsDetector.maxBatch) {
                       std::string emotion = emotionsDetector[i];
                       if (FLAGS_r) {
                       std::cout << "Predicted emotion = " << emotion << std::endl;
                       }
                       out << "," << emotion;
                       }
                       cv::putText(prev_frame,
                       out.str(),
                       cv::Point2f(result.location.x, result.location.y - 15),
                       cv::FONT_HERSHEY_COMPLEX_SMALL,
                       0.8,
                       cv::Scalar(0, 0, 255));
                       if (headPoseDetector.enabled() && i < headPoseDetector.maxBatch) {
                       if (FLAGS_r) {
                       std::cout << "Head pose results: yaw, pitch, roll = "
                       << headPoseDetector[i].angle_y << ";"
                       << headPoseDetector[i].angle_p << ";"
                       << headPoseDetector[i].angle_r << std::endl;
                       }
                       cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
                       headPoseDetector.drawAxes(prev_frame, center, headPoseDetector[i], 50);
                       }

                       if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
                       auto normed_landmarks = facialLandmarksDetector[i];
                       auto n_lm = normed_landmarks.size();
                       if (FLAGS_r)
                       std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
                       for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                       float normed_x = normed_landmarks[2 * i_lm];
                       float normed_y = normed_landmarks[2 * i_lm + 1];

                       if (FLAGS_r) {
                       std::cout << normed_x << ", "
                       << normed_y << std::endl;
                       }
                       int x_lm = rect.x + rect.width * normed_x;
                       int y_lm = rect.y + rect.height * normed_y;
                    // Drawing facial landmarks on the frame
                    cv::circle(prev_frame, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 255, 255), -1);
                    }
                    }
                    */

                    auto genderColor = (ageGenderDetector.enabled() && (i < ageGenderDetector.maxBatch)) ?
                        ((ageGenderDetector[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0, 0)) : cv::Scalar(100, 100, 100);
                    //cv::rectangle(prev_frame, result.location, genderColor, -1);
                    Rect clippedRect = result.location & cv::Rect(0, 0, width, height);
                    cv::Mat roi = prev_frame(clippedRect);
                    //cerr << boxes[i].x << ' ' << boxes[i].y << ' ' << boxes[i].width << ' ' << boxes[i].height << endl;
                    TrackingBox cur_box;
                    cur_box.box = clippedRect;
                    cur_box.id = i;
                    cur_box.frame = framesCounter-1;
                    detFrameData.push_back(cur_box);

                    //Mat roi = prev_frame(result.location);
                    //nlohmann::json obj = nlohmann::json::object({{"x", rect.x}, {"y", rect.y}, {"w", rect.width}, {"h", rect.height}});
                    //jObjects.push_back(obj);
                    redis_send = true;
                    doMosaic(roi, 10);
                    i++;
                }
                vector<TrackingBox> tracking_results = sorter.update(detFrameData);
                //cout << tracking_results.size() << endl;

//310319 Added the ifndef and #endif
#ifndef DEBUG
                for (TrackingBox it : tracking_results) {
                    nlohmann::json obj = \
                                         nlohmann::json::object({{"x", it.box.x}, {"y", it.box.y}, \
                                                 {"w", it.box.width}, {"h", it.box.height}, {"id", it.id}});
                    jObjects.push_back(obj);

                    rectangle(prev_frame, \
                            Point(it.box.x, it.box.y), Point(it.box.width+it.box.x, it.box.height+it.box.y), sorter.randColor[it.id % 20], 2,8,0);
                    //cv::putText(prev_frame, std::to_string(it.id), cv::Point(it.box.x, it.box.y), 0 ,1, cv::Scalar(0,0,0),3);
                }
#endif

#ifdef DEBUG
                //outtracking.write(prev_frame);
#endif


                j["rects"] = jObjects;
                if (redis_send) {
                    std::cout << "[ DEBUG ] json " << j.dump() << std::endl;
                    redisClient->send_faces(j.dump());
                    MatData matdata;
                    matdata.ir = prev_frame;

                    while(imagebuf.Size() >= 200 )
                        imagebuf.Pop_no_return();

                    imagebuf.Push(matdata);
                }

                //0504019 Removes the display of the OpenCV window to prevent the program from stopping when there is an input from the keyboard
                cv::imshow("Detection results", prev_frame);

                timer.finish("visualization");
            } else if (FLAGS_r) {
                // For every detected face
                for (int i = 0; i < prev_detection_results.size(); i++) {
                    if (ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch) {
                        out.str("");
                        out << (ageGenderDetector[i].maleProb > 0.5 ? "M" : "F");
                        out << std::fixed << std::setprecision(0) << "," << ageGenderDetector[i].age;
                        std::cout << "Predicted gender, age = " << out.str() << std::endl;
                    }

                    if (emotionsDetector.enabled() && i < emotionsDetector.maxBatch) {
                        std::cout << "Predicted emotion = " << emotionsDetector[i] << std::endl;
                    }

                    if (headPoseDetector.enabled() && i < headPoseDetector.maxBatch) {
                        std::cout << "Head pose results: yaw, pitch, roll = "
                            << headPoseDetector[i].angle_y << ";"
                            << headPoseDetector[i].angle_p << ";"
                            << headPoseDetector[i].angle_r << std::endl;
                    }

                    if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
                        auto normed_landmarks = facialLandmarksDetector[i];
                        auto n_lm = normed_landmarks.size();
                        std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
                        for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                            float normed_x = normed_landmarks[2 * i_lm];
                            float normed_y = normed_landmarks[2 * i_lm + 1];
                            std::cout << normed_x << ", " << normed_y << std::endl;
                        }
                    }
                }
            }
            // frame can hook here
            //outvideo.write(prev_frame);
            read_mosicimage_from_frame(prev_frame);

            // End of file (or a single frame file like an image). The last frame is displayed to let you check what is shown
            if (isLastFrame) {
                timer.finish("total");
                if (!FLAGS_no_wait) {
                    std::cout << "No more frames to process. Press any key to exit" << std::endl;
                    cv::waitKey(0);
                }
                break;
            } else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
                timer.finish("total");
                break;
            }

            prev_frame = frame;
            frame = next_frame;
            next_frame = cv::Mat();
        }
        //outvideo.release();

#ifdef DEBUG
        //outtracking.release();
#endif
        //rtsp_thread.join();

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Showing performance results
        if (FLAGS_pc) {
            faceDetector.printPerformanceCounts();
            ageGenderDetector.printPerformanceCounts();
            headPoseDetector.printPerformanceCounts();
            emotionsDetector.printPerformanceCounts();
            facialLandmarksDetector.printPerformanceCounts();
        }
        // ---------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
