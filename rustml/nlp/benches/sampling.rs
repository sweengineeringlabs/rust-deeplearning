use criterion::{Criterion, criterion_group, criterion_main, black_box};
use rustml_nlp::{argmax, apply_top_k, apply_top_p, apply_repetition_penalty, sample_categorical};

fn make_logits(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.01).sin()).collect()
}

fn bench_argmax(c: &mut Criterion) {
    let logits = make_logits(32000);
    c.bench_function("argmax_32k", |b| {
        b.iter(|| argmax(black_box(&logits)))
    });

    let logits = make_logits(128000);
    c.bench_function("argmax_128k", |b| {
        b.iter(|| argmax(black_box(&logits)))
    });
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k");
    for k in [10, 50, 100] {
        let logits = make_logits(32000);
        group.bench_function(format!("k={}_vocab=32k", k), |b| {
            b.iter_batched(
                || logits.clone(),
                |mut l| apply_top_k(black_box(&mut l), k),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_p");
    for p in [0.9_f32, 0.95, 0.99] {
        let logits = make_logits(32000);
        group.bench_function(format!("p={}_vocab=32k", p), |b| {
            b.iter_batched(
                || logits.clone(),
                |mut l| apply_top_p(black_box(&mut l), p),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let logits = make_logits(32000);
    let past_tokens: Vec<u32> = (0..100).collect();
    c.bench_function("rep_penalty_100tok_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut l| apply_repetition_penalty(black_box(&mut l), &past_tokens, 1.1),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_sample_categorical(c: &mut Criterion) {
    let logits = make_logits(32000);
    c.bench_function("sample_categorical_32k", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            sample_categorical(black_box(&logits), &mut rng)
        })
    });
}

criterion_group!(
    benches,
    bench_argmax,
    bench_top_k,
    bench_top_p,
    bench_repetition_penalty,
    bench_sample_categorical,
);
criterion_main!(benches);
