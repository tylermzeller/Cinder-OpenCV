#include "cinder/app/App.h"
#include "cinder/app/RendererGL.h"
#include "cinder/gl/gl.h"
#include "cinder/Capture.h"
#include "cinder/Log.h"

#include "CinderOpenCv.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class ocvFaceDetectApp : public App {
 public:
	void setup() override;
	void update() override;
	void draw() override;

    void updateFaces( Surface cameraImage );
    
private:
    void printDevices();


	CaptureRef			  mCapture;
	gl::TextureRef		mTexture;

	cv::CascadeClassifier	mFaceCascade, mEyeCascade;
	vector<Rectf>			mFaces, mEyes;

};

void ocvFaceDetectApp::setup()
{

	mFaceCascade.load( getAssetPath( "haarcascade_frontalface_alt.xml" ).string() );
	mEyeCascade.load( getAssetPath( "haarcascade_eye.xml" ).string() );

  printDevices();
  try {
    mCapture = Capture::create( 640, 480 );
  	mCapture->start();
  } catch ( ci::Exception &exc ) {
    CI_LOG_EXCEPTION( "Failed to init capture ", exc );
  }

}

void ocvFaceDetectApp::updateFaces( Surface cameraImage )
{
	const int calcScale = 2; // calculate the image at half scale

	// create a grayscale copy of the input image
	cv::Mat grayCameraImage( toOcv( cameraImage, CV_8UC1 ) );

	// scale it to half size, as dictated by the calcScale constant
	int scaledWidth = cameraImage.getWidth() / calcScale;
	int scaledHeight = cameraImage.getHeight() / calcScale;
	cv::Mat smallImg( scaledHeight, scaledWidth, CV_8UC1 );
	cv::resize( grayCameraImage, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR );

	// equalize the histogram
	cv::equalizeHist( smallImg, smallImg );

	// clear out the previously deteced faces & eyes
	mFaces.clear();
	mEyes.clear();

	// detect the faces and iterate them, appending them to mFaces
	vector<cv::Rect> faces;
	mFaceCascade.detectMultiScale( smallImg, faces );
	for( vector<cv::Rect>::const_iterator faceIter = faces.begin(); faceIter != faces.end(); ++faceIter ) {
		Rectf faceRect( fromOcv( *faceIter ) );
		faceRect *= calcScale;
		mFaces.push_back( faceRect );

		// detect eyes within this face and iterate them, appending them to mEyes
		vector<cv::Rect> eyes;
		mEyeCascade.detectMultiScale( smallImg( *faceIter ), eyes );
		for( vector<cv::Rect>::const_iterator eyeIter = eyes.begin(); eyeIter != eyes.end(); ++eyeIter ) {
			Rectf eyeRect( fromOcv( *eyeIter ) );
			eyeRect = eyeRect * calcScale + faceRect.getUpperLeft();
			mEyes.push_back( eyeRect );
		}
	}
}

void ocvFaceDetectApp::update()
{

#if defined( USE_HW_TEXTURE )
  if ( mCapture && mCapture->checkNewFrame() ) {
    mTexture = mCapture->getTexture();
  }
#else
  if ( mCapture && mCapture->checkNewFrame() ) {
    Surface surface = *mCapture->getSurface();
    if ( ! mTexture ) {
      // Capture images come back as top-down, and it's more efficient to keep them that way
      mTexture = gl::Texture::create( surface, gl::Texture::Format().loadTopDown() );
    }
    else {
      mTexture->update( surface );
    }
    updateFaces( surface );

  }
#endif
	// if( mCapture.checkNewFrame() ) {
	// 	Surface surface = mCapture.getSurface();
	// 	mCameraTexture = gl::Texture( surface );
	// 	updateFaces( surface );
	// }
}

void ocvFaceDetectApp::draw()
{

  gl::clear();

	if( ! mTexture )
		return;

	gl::setMatricesWindow( getWindowSize() );
	gl::enableAlphaBlending();

	// draw the webcam image
	gl::color( Color( 1, 1, 1 ) );
	gl::draw( mTexture );

	// draw the faces as transparent yellow rectangles
	gl::color( ColorA( 1, 1, 0, 0.45f ) );
	for( vector<Rectf>::const_iterator faceIter = mFaces.begin(); faceIter != mFaces.end(); ++faceIter )
		gl::drawSolidRect( *faceIter );

	// draw the eyes as transparent blue ellipses
	gl::color( ColorA( 0, 0, 1, 0.35f ) );
	for( vector<Rectf>::const_iterator eyeIter = mEyes.begin(); eyeIter != mEyes.end(); ++eyeIter )
		gl::drawSolidCircle( eyeIter->getCenter(), eyeIter->getWidth() / 2 );
}

void ocvFaceDetectApp::printDevices()
{
  for( const auto &device : Capture::getDevices() ) {
		console() << "Device: " << device->getName() << " "
#if defined( CINDER_COCOA_TOUCH ) || defined( CINDER_ANDROID )
		<< ( device->isFrontFacing() ? "Front" : "Rear" ) << "-facing"
#endif
		<< endl;
	}
}

void prepareSettings( ocvFaceDetectApp::Settings* settings )
{
#if defined( CINDER_ANDROID )
  settings->setKeepScreenOn( true );
#endif
}

CINDER_APP( ocvFaceDetectApp, RendererGl, prepareSettings )
