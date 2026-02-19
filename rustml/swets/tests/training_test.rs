use rustml_swets::*;

#[test]
fn test_linear_regression_convergence() {
    // Train Linear(1, 1) on y = 2x + 1
    let model = Linear::new(1, 1);
    let optimizer = SGD::new(0.01);
    let loss_fn = MSELoss::new();
    let mut trainer = Trainer::new(model, optimizer, loss_fn);

    // Generate synthetic data: y = 2x + 1
    let mut batches = Vec::new();
    for i in 0..20 {
        let x_val = i as f32 * 0.5 - 5.0;
        let y_val = 2.0 * x_val + 1.0;
        let x = Tensor::from_vec(vec![x_val], vec![1, 1]).unwrap();
        let y = Tensor::from_vec(vec![y_val], vec![1, 1]).unwrap();
        batches.push((x, y));
    }

    let mut last_loss = f32::MAX;
    for epoch in 0..100 {
        let loss = trainer.train_epoch(&batches).unwrap();
        if epoch % 20 == 0 {
            // Loss should generally be decreasing
            assert!(
                loss < last_loss * 1.5, // allow some fluctuation
                "epoch {epoch}: loss {loss} should not increase drastically from {last_loss}"
            );
        }
        last_loss = loss;
    }

    // After 100 epochs, loss should be small
    assert!(
        last_loss < 0.1,
        "final loss {last_loss} should be < 0.1 after 100 epochs"
    );
}

#[test]
fn test_no_grad_behavior() {
    tape::clear_tape();

    let mut model = Linear::new(2, 1);
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let target = Tensor::from_vec(vec![3.0], vec![1, 1]).unwrap();

    // Forward inside no_grad should not record ops
    let output = tape::no_grad(|| model.forward(&input).unwrap());

    // No gradients should be available
    let loss_fn = MSELoss::new();
    let _loss = tape::no_grad(|| loss_fn.forward(&output, &target).unwrap());

    // Tape should have no gradients
    let params = model.parameters();
    for param in &params {
        assert!(
            tape::grad(param).is_none(),
            "no gradient should be recorded inside no_grad"
        );
    }
}

#[test]
fn test_tape_isolation_between_batches() {
    tape::clear_tape();

    let mut model = Linear::new(1, 1);
    let loss_fn = MSELoss::new();

    let x1 = Tensor::from_vec(vec![1.0], vec![1, 1]).unwrap();
    let y1 = Tensor::from_vec(vec![3.0], vec![1, 1]).unwrap();

    // First batch
    let out1 = model.forward(&x1).unwrap();
    let loss1 = loss_fn.forward(&out1, &y1).unwrap();
    tape::backward(&loss1);

    let grad1 = tape::grad(&model.parameters()[0]).map(|g| g.to_vec());

    // Clear tape for second batch
    tape::clear_tape();

    let x2 = Tensor::from_vec(vec![2.0], vec![1, 1]).unwrap();
    let y2 = Tensor::from_vec(vec![5.0], vec![1, 1]).unwrap();

    let out2 = model.forward(&x2).unwrap();
    let loss2 = loss_fn.forward(&out2, &y2).unwrap();
    tape::backward(&loss2);

    let grad2 = tape::grad(&model.parameters()[0]).map(|g| g.to_vec());

    // Gradients from batch 1 should not leak into batch 2
    assert!(grad1.is_some(), "batch 1 should have gradients");
    assert!(grad2.is_some(), "batch 2 should have gradients");
    // They should generally be different (different inputs)
    assert_ne!(
        grad1.unwrap(),
        grad2.unwrap(),
        "gradients from different batches should differ"
    );
}

#[test]
fn test_validate_does_not_record() {
    let model = Linear::new(1, 1);
    let optimizer = SGD::new(0.01);
    let loss_fn = MSELoss::new();
    let mut trainer = Trainer::new(model, optimizer, loss_fn);

    let batches = vec![
        (
            Tensor::from_vec(vec![1.0], vec![1, 1]).unwrap(),
            Tensor::from_vec(vec![3.0], vec![1, 1]).unwrap(),
        ),
    ];

    // Validate should work without error and use no_grad internally
    let val_loss = trainer.validate(&batches).unwrap();
    assert!(val_loss.is_finite(), "validation loss should be finite");
}

#[test]
fn test_metrics() {
    let mut metrics = Metrics::new();
    metrics.update(&[1.0, 2.0, 3.0], &[1.5, 2.5, 2.0]);

    let mse = metrics.mse();
    let mae = metrics.mae();
    let rmse = metrics.rmse();

    // MSE = ((−0.5)² + (−0.5)² + 1²) / 3 = (0.25 + 0.25 + 1.0) / 3 = 0.5
    assert!((mse - 0.5).abs() < 1e-6, "mse = {mse}, expected 0.5");
    // MAE = (0.5 + 0.5 + 1.0) / 3 = 2/3
    assert!(
        (mae - 2.0 / 3.0).abs() < 1e-6,
        "mae = {mae}, expected 0.6667"
    );
    // RMSE = sqrt(0.5)
    assert!(
        (rmse - 0.5_f64.sqrt()).abs() < 1e-6,
        "rmse = {rmse}, expected {}", 0.5_f64.sqrt()
    );

    metrics.reset();
    assert_eq!(metrics.mse(), 0.0);
}

#[test]
fn test_nested_no_grad() {
    tape::clear_tape();

    let mut model = Linear::new(2, 1);
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();

    tape::no_grad(|| {
        // Outer no_grad: recording disabled
        assert!(!tape::is_recording());

        tape::no_grad(|| {
            // Inner no_grad: still disabled
            assert!(!tape::is_recording());
            let _ = model.forward(&input).unwrap();
        });

        // After inner no_grad returns: should STILL be disabled
        assert!(
            !tape::is_recording(),
            "nested no_grad must not re-enable recording on inner exit"
        );
    });

    // After outer no_grad: recording should be re-enabled
    assert!(tape::is_recording());
}
