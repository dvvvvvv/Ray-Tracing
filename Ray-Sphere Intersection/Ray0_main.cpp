#pragma once

#include "FL/Fl.H"
#include "FL/Fl_Double_Window.H"
#include "FL/Fl_RGB_Image.H"
#include <iostream>

#include "Ray0.h"


#define RESOLUTION_W 720
#define RESOLUTION_H 480


// RGB Color Buffer
uchar3 *h_color_buffer;



// Window for Display
class ImageWin : public Fl_Double_Window
{
public:

	// FLTK RGB Image
	Fl_RGB_Image *image_;

	ImageWin(int w, int h, const char *s=0) : Fl_Double_Window(w, h, s)
	{
		image_ = 0;
	};


	
	virtual void draw()
	{
		Fl_Double_Window::draw();

	
		// Draw the Image unless it's null
		if ( image_ != 0 ) image_->draw(0, 0);

	}

	
};



int main()
{
	
	h_color_buffer = new uchar3[RESOLUTION_W*RESOLUTION_H];

	


	BuildImage(h_color_buffer, RESOLUTION_W, RESOLUTION_H);


	
	// Create FLTK Window for displaying the result.
	ImageWin *win = new ImageWin(RESOLUTION_W, RESOLUTION_H);
	// Create FLTK RGB Image with my color buffer.
	win->image_ = new Fl_RGB_Image((unsigned char*)h_color_buffer, RESOLUTION_W, RESOLUTION_H, 3);
	win->resizable(win);
	win->end();
	win->show();

	return Fl::run();
}