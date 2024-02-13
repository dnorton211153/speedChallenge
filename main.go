//go:build ignore
// +build ignore

package main

import (
	"gocv.io/x/gocv"
	"log"
)

func main() {
	webcam, err := gocv.VideoCaptureDevice(0)
	if err != nil {
		log.Fatalf("error opening web cam: %v", err)
	}
	defer webcam.Close()

	img := gocv.NewMat()
	defer img.Close()

	window := gocv.NewWindow("webcamwindow")
	defer window.Close()

	for {
		if ok := webcam.Read(&img); !ok {
			log.Printf("cannot read device 0")
			return
		}
		if img.Empty() {
			continue
		}

		window.IMShow(img)
		window.WaitKey(50)

	}

}
