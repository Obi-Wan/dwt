/*
 * Exceptions.h
 *
 *  Created on: 7 oct. 2010
 *      Author: vigano
 */

#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <stdexcept>
#include <string>

using namespace std;

namespace dwt {

  class DwtBasicException : public exception {
    string message;

    exception nested_ex;

  public:
    DwtBasicException() { }
    DwtBasicException(const char * _mess) : message(_mess) { }
    DwtBasicException(const string& _mess) : message(_mess) { }

    DwtBasicException(const DwtBasicException & _ex)
          : exception(_ex), message(_ex.message), nested_ex(_ex.nested_ex) { }
    DwtBasicException(const string & _mess, const DwtBasicException& _ex)
          : message(_mess), nested_ex(_ex) { }

    virtual ~DwtBasicException() throw() { }

    virtual const char * what() const throw() { return message.c_str(); }
    const string& get_message() const throw() { return message; }
    const exception & get_nested_exception() const throw() { return nested_ex; }

    virtual void set_message(const string& _mess) { message = _mess; }
    virtual void append_message(const string& _mess) { message += _mess; }
    virtual void prefix_message(const string& _mess) { message = _mess + message; }
  };

  class DwtWrongArgumentException : public DwtBasicException {
  public:
    DwtWrongArgumentException() { }
    DwtWrongArgumentException(const DwtWrongArgumentException &_ex)
          : DwtBasicException(_ex) { }
    DwtWrongArgumentException(const char * _mess) : DwtBasicException(_mess) { }
    DwtWrongArgumentException(const string& _mess) : DwtBasicException(_mess) { }
    DwtWrongArgumentException(const string& _mess, const DwtBasicException& _ex)
          : DwtBasicException(_mess, _ex) { }
  };

  class DwtNotImplementedException : public DwtBasicException {
  public:
    DwtNotImplementedException() { }
    DwtNotImplementedException(const DwtNotImplementedException &_ex)
          : DwtBasicException(_ex) { }
    DwtNotImplementedException(const char * _mess) : DwtBasicException(_mess) { }
    DwtNotImplementedException(const string& _mess) : DwtBasicException(_mess) { }
    DwtNotImplementedException(const string& _mess, const DwtBasicException& _ex)
          : DwtBasicException(_mess, _ex) { }
  };

}  // namespace dwt


#endif /* EXCEPTIONS_H_ */
