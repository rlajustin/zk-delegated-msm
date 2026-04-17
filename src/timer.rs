use std::time::{Duration, Instant};

pub struct Timer {
    start_time: Instant,
    total_elapsed: Duration,
    is_paused: bool,
}
impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_elapsed: Duration::new(0, 0),
            is_paused: false,
        }
    }

    pub fn start(&mut self) {
        if self.is_paused {
            self.start_time = Instant::now();
            self.is_paused = false;
        }
    }

    pub fn pause(&mut self) {
        if !self.is_paused {
            self.total_elapsed += self.start_time.elapsed();
            self.is_paused = true;
        }
    }

    pub fn elapsed(&self) -> Duration {
        if self.is_paused {
            self.total_elapsed
        } else {
            self.total_elapsed + self.start_time.elapsed()
        }
    }
}
