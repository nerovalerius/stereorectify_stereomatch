#include <iostream>
#include <limits>
#include <cmath>
#include "opencv2/core.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/viz.hpp"
#include <opencv2/viz/types.hpp>
#include <opencv2/core/affine.hpp>

using namespace std;
using namespace cv;

int main(const int argc, const char * const argv[])
{


  // Argument check
  if(argc  < 5){
    std::cout << "please type in exactly 4 arguments" << std::endl;
    std::cout << "Usage: program <intrinsic-file> <extrinsic-file> <left-image> <right-image>" << std::endl;
    return -1;
  }


  // Parameter parsing
  String intrinsic_name(argv[1]);
  String extrinsic_name(argv[2]);
  auto left_input_file_name = argv[3];
  auto right_input_file_name = argv[4];

  // Mat Containers 
  Mat left_input_image, right_input_image, disparity_image, output_image, rectified_left_image, rectified_right_image, image_3d;

  // Intrinsic and Extrinsic matrices data structure
  FileStorage fs_intrinsics(intrinsic_name, FileStorage::READ);
  FileStorage fs_extrinsics(extrinsic_name, FileStorage::READ);



  // Read Images from file
  left_input_image = imread( left_input_file_name, IMREAD_COLOR );
  right_input_image = imread( right_input_file_name, IMREAD_COLOR );


  // Check if files could be opened
  if(!fs_intrinsics.isOpened())
  {
      printf("Failed to open file %s\n", intrinsic_name.c_str());
      return -1;
  }

  if(!fs_extrinsics.isOpened())
  {
      printf("Failed to open file %s\n", extrinsic_name.c_str());
      return -1;
  }



  // Intrinsics
  Mat M1, D1, M2, D2, M1_new, M2_new;
        fs_intrinsics["M1"] >> M1;
        fs_intrinsics["D1"] >> D1;
        fs_intrinsics["M2"] >> M2;
        fs_intrinsics["D2"] >> D2;


  // Extrinsics
  Mat R, T, R1, R2, P1, P2, Q;
        fs_extrinsics["R"] >> R;
        fs_extrinsics["T"] >> T;
        fs_extrinsics["R1"] >> R1;
        fs_extrinsics["R2"] >> R2;
        fs_extrinsics["P1"] >> P1;
        fs_extrinsics["P2"] >> P2;
        fs_extrinsics["Q"] >> Q;

  // Rectifying
  cv::Mat left_map1, left_map2, right_map1, right_map2;

  cv::stereoRectify(M1,D1,M2,D2, left_input_image.size(), R, T, R1, R2, P1, P2, Q);
  cv::initUndistortRectifyMap(M1, D1, R1, M1_new, left_input_image.size(),CV_32FC1, left_map1, left_map2);
  cv::initUndistortRectifyMap(M2, D2, R2, M2_new, right_input_image.size(),CV_32FC1, right_map1, right_map2);
  cv::remap(left_input_image, rectified_left_image, left_map1, left_map2, INTER_LINEAR);
  cv::remap(right_input_image, rectified_right_image, right_map1, right_map2, INTER_LINEAR);


  //  minDisparity      = 0
  //  numDisparities    = 80
  //  blockSize         = 1   - 1 = good value in this case - good quality
  //  P1                = 0   - See P2
  // 	P2                = 64  - controlling the disparity smoothness. Larger the values = smoother disparity.  >256 too blurred
  //  disp12MaxDiff     = 32  - better holefilling 
  //  preFilterCap      = 0   - greater values = worse 
  //  uniquenessRatio   = 0   - greater values = worse
  //  speckleWindowSize = 0   - greater values = worse
  //  speckleRange      = 0   - no speckle filtering - see 1 line above
  //  mode              !=    - StereoSGBM::MODE_HH - could not see any difference
  Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,80,1,0,64,32,0, 0,0);

  // Compute Disparity Image
  sgbm->compute(rectified_left_image, rectified_right_image, disparity_image);

  // Convert and scale image -> sgbm returns 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits
  // 4 fractional bits
  // 64 values = 6 Bit
  //
  //   00 0000 0000     6 bit + 4 bit       -> CONVERT TO
  //      0000 0000     8 bit               <-
  // == Bitshift * 2 >>
  //  >> 1 = * 0.5
  //  >> 2 = * 0.5 *0.5 = 0.25
  
  // disparity_image.convertTo(output_image,8,0.25); <- not scaled
 
  // Create 3D image
  //image_3d = left_input_image.clone();
  cv::reprojectImageTo3D(disparity_image, image_3d, Q);

  // iterate through 3d image
  image_3d.forEach<Vec3f>([&](Vec3f& pixel, const int* position) -> void {
        if( pixel[2] > 6 || pixel[2] < -4){
          pixel = std::numeric_limits<float>::quiet_NaN();
        }
  });

  // 3c cloud
  cv::viz::WCloud cloud(image_3d, rectified_left_image);

  // 3D Vizualizer Window
  cv::viz::Viz3d window_3d("3d_image");


  // Show a widget inside the 3D Window
  window_3d.showWidget("widget", cloud);




  // Start 3D window rendering
  window_3d.spinOnce(1,true);

  // Get Current Viewer Pose from 3d Window
  Affine3d currentPose = window_3d.getViewerPose();


  while(window_3d.wasStopped());


  // Get current rotation and translation
  Matx33d current_R = currentPose.rotation();
  Vec3d t = currentPose.translation();


  // Rotation Angle 180° 
  double alpha = CV_PI;

  // Rotation Matrix - Rotation around Z axis
  Matx33d rotationMatrix {
      cos(alpha),   -sin(alpha),     0,
      sin(alpha),    cos(alpha),     0,
               0,             0,     1,
  };

  // Make new Pose with new Rotation Matrix and old translation vector
  Affine3d newPose((rotationMatrix*current_R),t);

  // Set rotated pose at window
  window_3d.setViewerPose(newPose);

  // run vizualizer again
  window_3d.spin();


  // Wait for keystroke
  waitKey(0);



  return 0;
}


