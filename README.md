This is the OpenCV CinderBlock, designed for use with the open-source C++ library Cinder: http://libcinder.org

OpenCV 2.4.9.

#Installation (MacOS)
The preferred method of installing CinderOpenCv is to download the release from [cinder](https://libcinder.org/download).
The most current version of cinder that this repo supports is 0.9.1.

Unzip the release into a location you prefer. `cd` into the root folder:

```shell
workspace$ cd cinder_0.9.1_mac
workspace/cinder_0.9.1_mac$ ls
CMakeLists.txt	README.md	include		samples
COPYING		blocks		lib		src
docs		proj		tools
```

Clone this repo into your cinder root directory:

```shell
workspace/cinder_0.9.1_mac$ git clone https://github.com/tylermzeller/Cinder-OpenCV.git
workspace/cinder_0.9.1_mac$ ls
CMakeLists.txt	README.md	include		samples
COPYING		blocks		lib		src
Cinder-OpenCV	docs		proj		tools
```

You can now open the samples in xcode. Navigate to Cinder-OpenCV/samples/ocvFaceDetect/xcode
and open the .xcodeproj file. Click the play button to build and run. That's it!
