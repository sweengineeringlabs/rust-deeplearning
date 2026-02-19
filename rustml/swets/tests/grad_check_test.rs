use rustml_swets::*;

const EPS: f32 = 1e-3;
const REL_TOL: f32 = 1e-2; // relative tolerance for gradient checks

fn numerical_gradient<F>(f: F, input: &Tensor, eps: f32) -> Vec<f32>
where
    F: Fn(&Tensor) -> f32,
{
    let data = input.to_vec();
    let mut grads = vec![0.0f32; data.len()];

    for i in 0..data.len() {
        let mut plus_data = data.clone();
        plus_data[i] += eps;
        let plus_tensor = Tensor::from_vec(plus_data, input.shape().to_vec()).unwrap();
        let f_plus = f(&plus_tensor);

        let mut minus_data = data.clone();
        minus_data[i] -= eps;
        let minus_tensor = Tensor::from_vec(minus_data, input.shape().to_vec()).unwrap();
        let f_minus = f(&minus_tensor);

        grads[i] = (f_plus - f_minus) / (2.0 * eps);
    }

    grads
}

fn check_gradient(analytical: &[f32], numerical: &[f32], name: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{name}: gradient length mismatch"
    );
    for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let denom = a.abs().max(n.abs()).max(1e-8);
        let rel_err = (a - n).abs() / denom;
        assert!(
            rel_err < REL_TOL,
            "{name}[{i}]: analytical={a}, numerical={n}, rel_err={rel_err}"
        );
    }
}

#[test]
fn test_matmul_gradient() {
    tape::clear_tape();

    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    a.set_requires_grad(true);
    let mut b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    b.set_requires_grad(true);

    // Forward with tape
    use rustml_swets::api::tape::{TapeEntry, BackwardOp};

    let output = a.matmul_raw(&b).unwrap();
    tape::record_op(TapeEntry {
        backward_op: Box::new(rustml_swets::core::ops::matmul::MatMulBackward),
        output_id: output.id(),
        input_ids: vec![a.id(), b.id()],
        saved_tensors: vec![a.clone(), b.clone()],
    });

    let loss_val = output.sum_all_raw();
    let loss = Tensor::from_vec(vec![loss_val], vec![1]).unwrap();
    // Record a "sum" backward: gradient of sum is all ones
    struct SumBackward { shape: Vec<usize> }
    impl BackwardOp for SumBackward {
        fn backward(&self, _grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
            vec![Tensor::ones(self.shape.clone())]
        }
        fn name(&self) -> &str { "SumBackward" }
    }
    tape::record_op(TapeEntry {
        backward_op: Box::new(SumBackward { shape: output.shape().to_vec() }),
        output_id: loss.id(),
        input_ids: vec![output.id()],
        saved_tensors: vec![],
    });

    tape::backward(&loss);

    let grad_a = tape::grad(&a).expect("grad_a");
    let grad_b = tape::grad(&b).expect("grad_b");

    // Numerical gradients for a
    let b_clone = b.clone();
    let num_grad_a = numerical_gradient(
        |x| x.matmul_raw(&b_clone).unwrap().sum_all_raw(),
        &a,
        EPS,
    );
    check_gradient(&grad_a.to_vec(), &num_grad_a, "matmul_grad_a");

    // Numerical gradients for b
    let a_clone = a.clone();
    let num_grad_b = numerical_gradient(
        |x| a_clone.matmul_raw(x).unwrap().sum_all_raw(),
        &b,
        EPS,
    );
    check_gradient(&grad_b.to_vec(), &num_grad_b, "matmul_grad_b");
}

#[test]
fn test_add_gradient() {
    tape::clear_tape();

    use rustml_swets::api::tape::{TapeEntry, BackwardOp};

    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    a.set_requires_grad(true);
    let mut b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
    b.set_requires_grad(true);

    let output = a.add_raw(&b).unwrap();
    tape::record_op(TapeEntry {
        backward_op: Box::new(rustml_swets::core::ops::add::AddBackward {
            a_shape: a.shape().to_vec(),
            b_shape: b.shape().to_vec(),
        }),
        output_id: output.id(),
        input_ids: vec![a.id(), b.id()],
        saved_tensors: vec![],
    });

    let loss_val = output.sum_all_raw();
    let loss = Tensor::from_vec(vec![loss_val], vec![1]).unwrap();
    struct SumBackward { shape: Vec<usize> }
    impl BackwardOp for SumBackward {
        fn backward(&self, _grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
            vec![Tensor::ones(self.shape.clone())]
        }
        fn name(&self) -> &str { "SumBackward" }
    }
    tape::record_op(TapeEntry {
        backward_op: Box::new(SumBackward { shape: output.shape().to_vec() }),
        output_id: loss.id(),
        input_ids: vec![output.id()],
        saved_tensors: vec![],
    });

    tape::backward(&loss);

    let grad_a = tape::grad(&a).expect("grad_a").to_vec();
    let grad_b = tape::grad(&b).expect("grad_b").to_vec();

    // For sum(a + b), grad_a = grad_b = [1, 1, 1]
    assert_eq!(grad_a, vec![1.0, 1.0, 1.0]);
    assert_eq!(grad_b, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_mul_gradient() {
    tape::clear_tape();

    use rustml_swets::api::tape::{TapeEntry, BackwardOp};

    let mut a = Tensor::from_vec(vec![2.0, 3.0], vec![1, 2]).unwrap();
    a.set_requires_grad(true);
    let mut b = Tensor::from_vec(vec![4.0, 5.0], vec![1, 2]).unwrap();
    b.set_requires_grad(true);

    let output = a.mul_raw(&b).unwrap();
    tape::record_op(TapeEntry {
        backward_op: Box::new(rustml_swets::core::ops::mul::MulBackward),
        output_id: output.id(),
        input_ids: vec![a.id(), b.id()],
        saved_tensors: vec![a.clone(), b.clone()],
    });

    let loss_val = output.sum_all_raw();
    let loss = Tensor::from_vec(vec![loss_val], vec![1]).unwrap();
    struct SumBackward { shape: Vec<usize> }
    impl BackwardOp for SumBackward {
        fn backward(&self, _grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
            vec![Tensor::ones(self.shape.clone())]
        }
        fn name(&self) -> &str { "SumBackward" }
    }
    tape::record_op(TapeEntry {
        backward_op: Box::new(SumBackward { shape: output.shape().to_vec() }),
        output_id: loss.id(),
        input_ids: vec![output.id()],
        saved_tensors: vec![],
    });

    tape::backward(&loss);

    let grad_a = tape::grad(&a).expect("grad_a").to_vec();
    let grad_b = tape::grad(&b).expect("grad_b").to_vec();

    let b_clone = b.clone();
    let num_grad_a = numerical_gradient(
        |x| x.mul_raw(&b_clone).unwrap().sum_all_raw(),
        &a,
        EPS,
    );
    check_gradient(&grad_a, &num_grad_a, "mul_grad_a");

    let a_clone = a.clone();
    let num_grad_b = numerical_gradient(
        |x| a_clone.mul_raw(x).unwrap().sum_all_raw(),
        &b,
        EPS,
    );
    check_gradient(&grad_b, &num_grad_b, "mul_grad_b");
}

#[test]
fn test_relu_gradient() {
    tape::clear_tape();

    use rustml_swets::api::tape::{TapeEntry, BackwardOp};
    use rustml_swets::core::ops::relu::ReLUBackward;

    let mut input = Tensor::from_vec(vec![-1.0, 0.5, -0.3, 2.0], vec![2, 2]).unwrap();
    input.set_requires_grad(true);

    let output = input.relu_raw();
    tape::record_op(TapeEntry {
        backward_op: Box::new(ReLUBackward),
        output_id: output.id(),
        input_ids: vec![input.id()],
        saved_tensors: vec![input.clone()],
    });

    let loss_val = output.sum_all_raw();
    let loss = Tensor::from_vec(vec![loss_val], vec![1]).unwrap();
    struct SumBackward { shape: Vec<usize> }
    impl BackwardOp for SumBackward {
        fn backward(&self, _grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
            vec![Tensor::ones(self.shape.clone())]
        }
        fn name(&self) -> &str { "SumBackward" }
    }
    tape::record_op(TapeEntry {
        backward_op: Box::new(SumBackward { shape: output.shape().to_vec() }),
        output_id: loss.id(),
        input_ids: vec![output.id()],
        saved_tensors: vec![],
    });

    tape::backward(&loss);

    let grad = tape::grad(&input).expect("grad").to_vec();
    // ReLU grad: 0 for negative, 1 for positive
    assert_eq!(grad, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_mse_gradient() {
    tape::clear_tape();

    let mut pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    pred.set_requires_grad(true);
    let target = Tensor::from_vec(vec![1.5, 2.5, 2.0], vec![3]).unwrap();

    let loss_fn = MSELoss::new();
    let loss = loss_fn.forward(&pred, &target).unwrap();
    tape::backward(&loss);

    let grad = tape::grad(&pred).expect("grad").to_vec();

    // Numerical gradient
    let target_clone = target.clone();
    let num_grad = numerical_gradient(
        |x| {
            let diff = x.sub_raw(&target_clone).unwrap();
            let sq = diff.pow_raw(2.0);
            sq.mean_all_raw()
        },
        &pred,
        EPS,
    );
    check_gradient(&grad, &num_grad, "mse_grad");
}

#[test]
fn test_linear_gradient() {
    tape::clear_tape();

    let mut linear = Linear::new(3, 2);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let target = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]).unwrap();

    let output = linear.forward(&input).unwrap();
    let loss_fn = MSELoss::new();
    let loss = loss_fn.forward(&output, &target).unwrap();
    tape::backward(&loss);

    // Verify gradients exist for weight and bias
    let params = linear.parameters();
    for param in &params {
        let grad = tape::grad(param);
        assert!(grad.is_some(), "gradient should exist for parameter");
        let grad = grad.unwrap();
        assert_eq!(grad.shape(), param.shape(), "gradient shape should match parameter shape");
    }
}
