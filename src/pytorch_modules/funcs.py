import torch
from permutohedral_encoding.pytorch_modules.find_cpp_package import *

_C=find_package()



class PermutoEncodingFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, lattice, lattice_values, positions, anneal_window, require_lattice_values_grad, require_positions_grad):

		#forward
		input_struct=_C.EncodingInput(lattice_values, positions, anneal_window, require_lattice_values_grad, require_positions_grad)
		sliced_values=lattice.forward(input_struct )

		#save for back
		ctx.lattice=lattice
		ctx.input_struct=input_struct
		ctx.save_for_backward(sliced_values)


		return sliced_values

	@staticmethod
	def backward(ctx, grad_sliced_values_monolithic):

		#restore from ctx
		lattice=ctx.lattice
		input_struct=ctx.input_struct
		sliced_values,  =ctx.saved_tensors

		assert input_struct.m_require_lattice_values_grad or input_struct.m_require_positions_grad, "We cannot perform the backward function on the slicing because we did not precompute the required tensors in the forward pass. To enable this, set the model.train(), set torch.set_grad_enabled(True) and make lattice_values have required_grad=True"
	  
		#we pass the tensors of lattice_values and positiosn explicitly and not throught the input struct so that we can compute gradients from them for the double backward pass
		return PermutoEncodingFuncBack.apply(lattice, input_struct, grad_sliced_values_monolithic, input_struct.m_lattice_values, input_struct.m_positions_raw, sliced_values) 
	   

# in order to enable a double backward like in https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class PermutoEncodingFuncBack(torch.autograd.Function):
	@staticmethod
	def forward(ctx, lattice, input_struct, grad_sliced_values_monolithic, lattice_values, positions, sliced_values_hom):

		lattice_values_grad=None
		positions_grad=None
		
		if input_struct.m_require_lattice_values_grad or input_struct.m_require_positions_grad:
			grad_sliced_values_monolithic=grad_sliced_values_monolithic.contiguous()

			ctx.save_for_backward(grad_sliced_values_monolithic)
			ctx.lattice=lattice
			ctx.input_struct=input_struct
			
			lattice_values_grad, positions_grad=lattice.backward(input_struct, grad_sliced_values_monolithic) 

		return None, lattice_values_grad, positions_grad, None, None, None
	@staticmethod
	def backward(ctx, dummy1, double_lattice_values_grad, double_positions_grad, dummy5, dummy6, dummy7):

		#in the forward pass of this module we do 
		#lattice_values_grad, positions_grad = slice_back(lattice_values_monolithic, grad_sliced_values_monolithic, positions)
		#now in the backward pass we have the upstream gradient which is double_lattice_values_grad, double_positions_grad
		#we want to propagate the double_positions_grad into lattice_values_monolithic and grad_sliced_values_monolithic

		grad_sliced_values_monolithic, =ctx.saved_tensors
		lattice=ctx.lattice
		input_struct=ctx.input_struct


		grad_lattice_values_monolithic, grad_grad_sliced_values_monolithic=lattice.double_backward_from_positions(input_struct, double_positions_grad,grad_sliced_values_monolithic )

		
		return None, None, grad_grad_sliced_values_monolithic, grad_lattice_values_monolithic, None, None, None, None, None, None, None, None, None, None




