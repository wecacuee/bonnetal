/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */

// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>

// net stuff
#include <selector.hpp>
namespace cl = bonnetal::segmentation;

// standalone lib h
#include "infer.hpp"

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

int main(int argc, const char *argv[]) {
  // define options
  std::string video = "";
  std::string path;
  std::string backend = "pytorch";
  bool verbose = false;

  // Parse options
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")(
        "video", po::value<std::string>(),
        "Video to infer. Defaults to webcam.")(
        "path,p", po::value<std::string>(),
        "Directory to get the inference model from. No default")(
        "backend,b", po::value<std::string>(),
        "Backend. Pytorch, and TensorRT.")(
        "verbose,v", po::bool_switch(),
        "Verbose mode. Calculates profile (time to run)");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    if (vm.count("video")) {
      video = vm["video"].as<std::string>();
      std::cout << "video: " << video << std::endl;
    } else {
      std::cout << "video: Using default (camera)!" << std::endl;
    }

    // make defaults count, parameter check, and print
    if (vm.count("path")) {
      path = vm["path"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "path: " << path << std::endl;
    } else {
      std::cerr << "No path! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }
    if (vm.count("backend")) {
      backend = vm["backend"].as<std::string>();
      std::cout << "backend: " << backend << std::endl;
    } else {
      std::cout << "backend: " << backend << ". Using default!" << std::endl;
    }
    if (vm.count("verbose")) {
      verbose = vm["verbose"].as<bool>();
      std::cout << "verbose: " << verbose << std::endl;
    } else {
      std::cout << "verbose: " << verbose << ". Using default!" << std::endl;
    }

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  } catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // create a network
  std::unique_ptr<cl::Net> net = cl::make_net(path, backend);

  // set verbosity
  net->verbosity(verbose);

  // open capture
  std::unique_ptr<cv::VideoCapture> cap;
  std::unique_ptr<cv::VideoWriter> color_writer;
  std::unique_ptr<cv::VideoWriter> blend_writer;
  if (video == "") {
    std::cout << "Opening webcam for prediction." << std::endl;
    cap = std::unique_ptr<cv::VideoCapture>(new cv::VideoCapture(0));
  } else {
    std::cout << "Opening video" << video << " for prediction." << std::endl;
    cap = std::unique_ptr<cv::VideoCapture>(new cv::VideoCapture(video));
  }
  if (!cap->isOpened())  // check if we succeeded
  {
    return 1;
  }

  if (verbose) {
    cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Blend", cv::WINDOW_AUTOSIZE);
  }

  // predict each image
  for (int i = 0;; i++) {
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
    std::cout << "Predicting frame: " << i << std::endl;

    // Open a frame
    cv::Mat frame;
    *cap >> frame;  // get a new frame from camera
    // Check for invalid input
    if (!frame.data) {
      std::cerr << "No image in frame!" << std::endl;
      return 1;
    }

    // predict
    cv::Mat argmax = net->infer(frame);

    // get color
    cv::Mat color_mask = net->color(argmax);

    // get color
    cv::Mat blend_mask = net->blend(frame, color_mask);

    // print the output
    if (verbose) {
      cv::imshow("Frame", frame);       // Show our image inside
      cv::imshow("Mask", color_mask);   // Show our image inside
      cv::imshow("Blend", blend_mask);  // Show our image inside
      cv::waitKey(1);
    }


    if (video != "" && ( !  (color_writer && blend_writer))) {
      fs::path color_videop(video);
      std::string orig_ext = color_videop.extension().string();
      color_videop.replace_extension(".segmentation-color" + orig_ext);
      color_writer = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
          color_videop.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
          /*fps=*/cap->get(CV_CAP_PROP_FPS), color_mask.size(),
          /*isColor=*/true));

      fs::path blend_videop(video);
      blend_videop.replace_extension(".segmentation-blend" + orig_ext);
      blend_writer = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
          blend_videop.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
          /*fps=*/cap->get(CV_CAP_PROP_FPS), blend_mask.size(),
          /*isColor=*/true));
    }
    if (color_writer->isOpened()) {
      color_writer->write(color_mask);
    }
    if (blend_writer->isOpened()) {
      blend_writer->write(blend_mask);
    }
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  }

  return 0;
}
