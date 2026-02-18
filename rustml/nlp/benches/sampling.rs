use criterion::{Criterion, criterion_group, criterion_main, black_box};
use rustml_nlp::{
    argmax, apply_top_k, apply_top_p, apply_top_p_buffered, apply_repetition_penalty,
    sample_categorical, SamplingBuffer,
};

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

fn bench_top_p_buffered(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_p_buffered_vs_alloc");
    let logits = make_logits(32000);
    let p = 0.9f32;

    // Allocating apply_top_p (baseline)
    group.bench_function("top_p_alloc_p=0.9_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut l| apply_top_p(black_box(&mut l), p),
            criterion::BatchSize::SmallInput,
        )
    });

    // Buffered apply_top_p_buffered (reuses sort buffer)
    let mut sort_buf: Vec<(f32, usize)> = Vec::with_capacity(32000);
    group.bench_function("top_p_buffered_p=0.9_32k", |b| {
        b.iter_batched(
            || logits.clone(),
            |mut l| apply_top_p_buffered(black_box(&mut l), p, &mut sort_buf),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_full_sampling_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_pipeline");
    let logits = make_logits(32000);
    let past_tokens: Vec<u32> = (0..100).collect();
    let temperature = 0.8f32;
    let top_k = 50usize;
    let top_p = 0.9f32;
    let rep_penalty = 1.1f32;

    // Allocating pipeline (per-token allocation pattern)
    group.bench_function("pipeline_alloc_32k", |b| {
        b.iter(|| {
            let mut l = logits.clone(); // alloc 1
            apply_repetition_penalty(&mut l, &past_tokens, rep_penalty);
            for v in l.iter_mut() { *v /= temperature; }
            apply_top_k(&mut l, top_k);
            apply_top_p(&mut l, top_p); // alloc 2 (sort vec)
            let mut rng = rand::thread_rng();
            black_box(sample_categorical(&l, &mut rng))
        })
    });

    // Buffered pipeline (reuses pre-allocated buffers)
    let mut buf = SamplingBuffer::new(32000);
    group.bench_function("pipeline_buffered_32k", |b| {
        b.iter(|| {
            buf.logits.clear();
            buf.logits.extend_from_slice(&logits); // reuse capacity
            apply_repetition_penalty(&mut buf.logits, &past_tokens, rep_penalty);
            for v in buf.logits.iter_mut() { *v /= temperature; }
            apply_top_k(&mut buf.logits, top_k);
            apply_top_p_buffered(&mut buf.logits, top_p, &mut buf.sort_buf); // reuse sort buf
            let mut rng = rand::thread_rng();
            black_box(sample_categorical(&buf.logits, &mut rng))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_argmax,
    bench_top_k,
    bench_top_p,
    bench_repetition_penalty,
    bench_sample_categorical,
    bench_top_p_buffered,
    bench_full_sampling_pipeline,
);
criterion_main!(benches);
