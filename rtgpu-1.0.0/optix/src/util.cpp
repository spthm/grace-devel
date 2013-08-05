#include "pch.h"

void check_glerror()
{
	std::ostringstream errors;
	GLenum err;
	while((err = glGetError()) != GL_NO_ERROR)
		errors << (const char *)gluErrorString(err) << '\n';

	if(!errors.str().empty())
		throw std::runtime_error("OpenGL: "+errors.str());
};
