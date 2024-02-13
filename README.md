# SpeedTest with Laneline Detection (AI)
@Author: [David Norton](https://github.com/dnorton211153)

## Overview

Nothing super-serious here, but I wanted to prove some concepts,
and this is the result.  I wanted to see if I could detect lanelines
in a video, and then superimpose the lanelines on the video.  I also
wanted to see if I could detect the speed of the vehicle in the video,
and then superimpose the speed on the video.  I also wanted to see if
I could do all of this in a web application, and then display the
output video in a web browser.

I used Keras/Tensorflow/OpenCV for the laneline and speed detection,
and the web application runs in a Django web container (WSGI).
Should also work in a Docker container, etc., but I haven't tested.

I'm not going to go into the details of the code here, but I will
provide a high-level overview of the code.

Unfinished GoLang code is also included, but it's not complete yet.

======

## High-Level Overview

[speedTest] is the python web application, which runs in a Django container (WSGI).

The AI (keras/tensorflow/cv2) includes the Speed Test and Laneline Detection.  

After the end user submits his MP4/similar video via the web form, the code in [speedTest](https://github.com/dnorton211153/speedChallenge/blob/main/speedTest/speedTest/speedTestWithLanelines.py) will superimpose the detected Lanelines and speed predictions, upon the video.

Then it will display the resulting video to the user, in the response.

I'm accessing the webapp from a standard web browser; Chrome 121.0.6167.161 (Official Build) (64-bit) on Windows 10 is working well.

======
