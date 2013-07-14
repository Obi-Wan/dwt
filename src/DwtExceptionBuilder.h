/*
 * DwtExceptionBuilder.h
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#ifndef DWTEXCEPTIONBUILDER_H_
#define DWTEXCEPTIONBUILDER_H_

#include "dwt_definitions.h"
#include "dwt_exceptions.h"

#include <sstream>

namespace dwt {

  class DwtExceptionBuilder {
  protected:
    template<typename FirstType, typename ... Args>
    void
    add_arg(stringstream & stream, FirstType first, Args ... args);
    void
    add_arg(stringstream & stream) { }
  public:
    DwtExceptionBuilder() = default;
    ~DwtExceptionBuilder() = default;

    template<class ExceptionType, typename ... Args>
    ExceptionType
    build(Args... args);
  };

} /* namespace dwt */

template<class ExceptionType, typename ... Args>
ExceptionType
dwt::DwtExceptionBuilder::build(Args ... args)
{
  stringstream stream;
  add_arg(stream, args ...);

  return ExceptionType(stream.str());
}

template<typename FirstType, typename ... Args>
void
dwt::DwtExceptionBuilder::add_arg(stringstream & stream, FirstType first, Args ... args)
{
  stream << first;
  add_arg(stream, args ...);
}

#endif /* DWTEXCEPTIONBUILDER_H_ */
