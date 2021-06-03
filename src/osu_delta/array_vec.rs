pub trait ArrayVec: Sized {
    fn neg(&mut self) -> &mut Self;
    fn scalar_mult(&mut self, scalar: f32) -> &mut Self;
    fn pointwise_exp(&mut self) -> &mut Self;
    fn powi_mean(&self, pow: i32) -> f32;
    fn powf_mean(&self, pow: f32) -> f32;
    fn pointwise_add(&mut self, other: &Self) -> &mut Self;
    fn pointwise_mult(&mut self, other: &Self) -> &mut Self;
    fn pointwise_powf(&mut self, pow: f32) -> &mut Self;
}

impl<const N: usize> ArrayVec for [f32; N] {
    fn neg(&mut self) -> &mut Self {
        for elem in self.iter_mut() {
            *elem *= -1.0;
        }

        self
    }

    fn scalar_mult(&mut self, scalar: f32) -> &mut Self {
        for elem in self.iter_mut() {
            *elem *= scalar;
        }

        self
    }

    fn pointwise_exp(&mut self) -> &mut Self {
        for elem in self.iter_mut() {
            *elem = elem.exp();
        }

        self
    }

    fn powi_mean(&self, pow: i32) -> f32 {
        let mut sum = 0.0;

        for elem in self.iter() {
            sum += elem.powi(pow);
        }

        (sum / self.len() as f32).powf(1.0 / pow as f32)
    }

    fn powf_mean(&self, pow: f32) -> f32 {
        let mut sum = 0.0;

        for elem in self.iter() {
            sum += elem.powf(pow);
        }

        (sum / self.len() as f32).powf(1.0 / pow as f32)
    }

    fn pointwise_add(&mut self, other: &Self) -> &mut Self {
        for (elem, term) in self.iter_mut().zip(other.iter()) {
            *elem += term;
        }

        self
    }

    fn pointwise_mult(&mut self, other: &Self) -> &mut Self {
        for (elem, factor) in self.iter_mut().zip(other.iter()) {
            *elem *= factor;
        }

        self
    }

    fn pointwise_powf(&mut self, pow: f32) -> &mut Self {
        for elem in self.iter_mut() {
            *elem = elem.powf(pow);
        }

        self
    }
}
