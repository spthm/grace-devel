#include "pch.h"
#include "gui.h"
#include "canvas.h"

class main_frame : public MainFrameUI
{
public:
    main_frame();

    void on_exit();

    void redraw_canvas();

private:
    s3d::fps_counter m_fps;

    canvas *m_canvas;
};

main_frame::main_frame()
    : MainFrameUI(0,0,800,600,"rodlima - RTGPU - PMPP/2010")
{
    m_canvas = new canvas(0,30,w(), h()-50);
    add_resizable(*m_canvas);
}

void main_frame::redraw_canvas()
{
    if(m_fps.update(0.5))
    {
	static char buf[50];
	snprintf(buf,sizeof(buf),"FPS: %.2f",m_fps.value());
	m_status->label(buf);
    }

    m_canvas->redraw();
}

void main_frame::on_exit()/*{{{*/
{
    hide();
}/*}}}*/

/*{{{ Proxy event handlers */
#define CATCH() \
    catch(std::exception &e) \
{ \
    fl_alert("%s",e.what()); \
} \
catch(...) \
{ \
    fl_alert("Unknown error"); \
}

namespace {
    main_frame *get_frame(Fl_Widget *w)
    {
	if(auto *frame = dynamic_cast<main_frame *>(w))
	    return frame;
	else
	{
	    assert(w);
	    return get_frame(w->parent());
	}
    }
}

void on_file_exit(Fl_Menu_ *m, void *)/*{{{*/
{
    try
    {
	get_frame(m)->on_exit();
    }
    CATCH()
}/*}}}*/
/*}}}*/

int main(int argc, char *argv[])
{
    try
    {
	main_frame frm;
	frm.show();


	while(Fl::first_window() != NULL)
	{
	    Fl::wait(0);//1/75.0);
	    frm.redraw_canvas();
	}

	return 0;
    }
    catch(std::exception &e)
    {
	std::cerr << e.what() << std::endl;
	return 1;
    }
}
