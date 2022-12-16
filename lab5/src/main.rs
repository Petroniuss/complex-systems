#![allow(non_snake_case)]

use std::collections::HashMap;
use std::fs;
use rand::{Rng, thread_rng};
use tqdm::{Iter};
use average::{Mean};

fn main() {
    // simulation_1();
    // simulation_2();
    // simulation_3();
    // simulation_4();
    // simulation_5();
    // simulation_6();
    // simulation_7();
    // simulation_8();
    // simulation_9();

    // simulation_1_4_agents();
    // simulation_2_4_agents();
    // simulation_3_4_agents();
    simulation_6_4_agents();
    simulation_7_4_agents();
}

fn simulation_1() {
    simulation(
        3000, 20, 0.5, 4, 2,
        "results/simulation_1_before.txt",
        "results/simulation_1_after.txt",
        "results/simulation_1_stats.txt",
    );
}

fn simulation_2() {
    simulation(
        3000, 100, 0.5, 4, 2,
        "results/simulation_2_before.txt",
        "results/simulation_2_after.txt",
        "results/simulation_2_stats.txt",
    );
}

fn simulation_3() {
    simulation(
        3000, 800, 0.5, 4, 2,
        "results/simulation_3_before.txt",
        "results/simulation_3_after.txt",
        "results/simulation_3_stats.txt",
    );
}


fn simulation_1_4_agents() {
    simulation(
        3000, 20, 0.5, 4, 4,
        "results/simulation_1_4_agents_before.txt",
        "results/simulation_1_4_agents_after.txt",
        "results/simulation_1_4_agents_stats.txt",
    );
}

fn simulation_2_4_agents() {
    simulation(
        3000, 100, 0.5, 4, 4,
        "results/simulation_2_4_agents_before.txt",
        "results/simulation_2_4_agents_after.txt",
        "results/simulation_2_4_agents_stats.txt",
    );
}

fn simulation_3_4_agents() {
    simulation(
        3000, 200, 0.5, 4, 4,
        "results/simulation_3_4_agents_before.txt",
        "results/simulation_3_4_agents_after.txt",
        "results/simulation_3_4_agents_stats.txt",
    );
}

fn simulation_6_4_agents() {
    simulation(
        3000, 200, 0.65, 4, 4,
        "results/simulation_6_4_agents_before.txt",
        "results/simulation_6_4_agents_after.txt",
        "results/simulation_6_4_agents_stats.txt",
    );
}

fn simulation_7_4_agents() {
    simulation(
        3000, 20, 0.35, 4, 4,
        "results/simulation_7_4_agents_before.txt",
        "results/simulation_7_4_agents_after.txt",
        "results/simulation_7_4_agents_stats.txt",
    );
}

fn simulation_4() {
    simulation(
        3000, 20, 0.65, 4, 2,
        "results/simulation_4_before.txt",
        "results/simulation_4_after.txt",
        "results/simulation_4_stats.txt",
    );
}

fn simulation_5() {
    simulation(
        3000, 100, 0.65, 4, 2,
        "results/simulation_5_before.txt",
        "results/simulation_5_after.txt",
        "results/simulation_5_stats.txt",
    );
}

fn simulation_6() {
    simulation(
        3000, 200, 0.65, 4, 2,
        "results/simulation_6_before.txt",
        "results/simulation_6_after.txt",
        "results/simulation_6_stats.txt",
    );
}

fn simulation_7() {
    simulation(
        3000, 20, 0.35, 4, 2,
        "results/simulation_7_before.txt",
        "results/simulation_7_after.txt",
        "results/simulation_7_stats.txt",
    );
}

fn simulation_8() {
    simulation(
        3000, 100, 0.35, 4, 2,
        "results/simulation_8_before.txt",
        "results/simulation_8_after.txt",
        "results/simulation_8_stats.txt",
    );
}

fn simulation_9() {
    simulation(
        3000, 200, 0.35, 4, 2,
        "results/simulation_9_before.txt",
        "results/simulation_9_after.txt",
        "results/simulation_9_stats.txt",
    );
}


fn simulation(n_iters: usize,
              L: usize,
              p: f32,
              alpha: usize,
              num_agents: i32,
              before_file_name: &str,
              after_file_name: &str,
              after_statistics_file_name: &str,
) {

    let mut M = automata(L, p, num_agents);
    save_matrix(&M, before_file_name);

    let mut S = Vec::with_capacity(n_iters);
    for _ in (0..n_iters).into_iter().tqdm() {
        let mean_s = simulation_iteration(&mut M, alpha);
        S.push(mean_s);
    }

    save_matrix(&M, after_file_name);
    save_vec(&S, after_statistics_file_name);
}

fn simulation_iteration(M: &mut Vec<Vec<i32>>, alpha: usize) -> f64 {
    // could be done without allocation but let's keep it.
    fn neighbours(n: usize, j: usize, i: usize) -> Vec<(usize, usize)> {
        let j = j as i32;
        let i = i as i32;
        let n = n as i32;
        let mut indices = Vec::with_capacity(8);
        for nj in (j - 1)..=(j + 1) {
            for ni in (i - 1)..=(i + 1) {
                if nj != ni && ni != 0 {
                    let nj = nj.rem_euclid(n) as usize;
                    let ni = ni.rem_euclid(n) as usize;
                    indices.push((nj, ni));
                }
            }
        }

        return indices;
    }

    fn count_neighbours(j: usize, i: usize, M: &Vec<Vec<i32>>) -> usize {
        let n = M.len();
        let agent_type = M[j][i];
        let mut count = 0;
        for (nj, ni) in neighbours(n, j, i) {
            if M[nj][ni] == agent_type {
                count += 1;
            }
        }

        return count;
    }

    fn pick_empty_spot(j: usize, i: usize, M: &Vec<Vec<i32>>) -> Option<(usize, usize)> {
        let n = M.len();
        let empty_spots = neighbours(n, j, i)
            .into_iter()
            .filter(|(nj, ni)| M[*nj][*ni] == 0)
            .collect::<Vec<_>>();


        if empty_spots.is_empty() {
            return None;
        }

        let mut rng = thread_rng();
        let idx = rng.gen_range(0..empty_spots.len());

        return Some(empty_spots[idx]);
    }

    let n = M.len();
    let mut state = HashMap::new();
    let mut S = Vec::with_capacity(n);
    for j in 0..n {
        for i in 0..n {
            let agent_type = M[j][i];
            if agent_type > 0 {
                // todo compute mean s.
                let s = count_neighbours(j, i, &M);
                S.push(s);
                if s < alpha {
                    if let Some(empty_spot) = pick_empty_spot(j, i, &M) {
                        state.entry(empty_spot)
                            .or_insert(Vec::new())
                            .push((j, i));
                    }
                }
            }
        }
    }

    for ((nj, ni), old_positions) in state.into_iter() {
        if old_positions.len() > 1 {
            continue;
        }

        let (j, i) = old_positions[0];
        M[nj][ni] = M[j][i];
        M[j][i] = 0;
    }

    S.into_iter().map(|x| x as f64).collect::<Mean>().mean()
}


fn save_matrix(M: &Vec<Vec<i32>>, filename: &str) {
    let data = M.into_iter()
        .map(|x|
            x.into_iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ).collect::<Vec<_>>()
        .join("\n");

    fs::write(filename, data).expect("");
}

fn save_vec(M: &Vec<f64>, filename: &str) {
    let data = M.into_iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(" ");

    fs::write(filename, data).expect("");
}

fn automata(L: usize, p: f32, num_agents: i32) -> Vec<Vec<i32>> {
    let mut M = vec![vec![0; L]; L];
    let size = L * L;

    let mut taken = 0;
    let taken_limit = (p * size as f32) as i32;
    let mut rng = thread_rng();
    while taken < taken_limit {
        let x = rng.gen_range(0..L);
        let y = rng.gen_range(0..L);

        if M[y][x] == 0 {
            M[y][x] = rng.gen_range(1..=num_agents);
            taken += 1;
        }
    }

    return M;
}
