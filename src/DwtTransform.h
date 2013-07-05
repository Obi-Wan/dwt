/*
 * DwtTransform.h
 *
 *  Created on: Jul 5, 2013
 *      Author: vigano
 */

#ifndef DWTTRANSFORM_H_
#define DWTTRANSFORM_H_

#include "dwt_definitions.h"

#include "DwtVolume.h"

namespace dwt {

	template<typename Type>
	class DwtTransform {
	public:
		DwtTransform();
		virtual ~DwtTransform();
	};

} /* namespace dwt */
#endif /* DWTTRANSFORM_H_ */
