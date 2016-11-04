import tensorflow as tf
from tensorflow.python.training.optimizer import Optimizer


class ClippingRMSPropOptimizer(tf.train.RMSPropOptimizer):
    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=0.1,
                 clip_norm=40.0,
                 name="ClippingRMSProp",
                 use_locking=False
                 ):
        super(ClippingRMSPropOptimizer, self).__init__(learning_rate,
                                                       decay,
                                                       momentum,
                                                       epsilon,
                                                       use_locking,
                                                       name)
        self._clip_norm = clip_norm

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        grads_and_vars = super(ClippingRMSPropOptimizer, self).compute_gradients(loss, var_list=var_list,
                                                                                 gate_gradients=gate_gradients,
                                                                                 aggregation_method=aggregation_method,
                                                                                 colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                                                 grad_loss=grad_loss)
        if self._clip_norm is not None and self._clip_norm > 0:
            grads_and_vars = [(tf.clip_by_norm(grad, self._clip_norm), var) for grad, var in grads_and_vars]
        return grads_and_vars

    def _apply_dense(self, grad, var):
        """'Implemented' for the sake of PyCharm Warnings"""
        return super(ClippingRMSPropOptimizer, self)._apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        """'Implemented' for the sake of PyCharm Warnings"""
        return super(ClippingRMSPropOptimizer, self)._apply_sparse(grad, var)
