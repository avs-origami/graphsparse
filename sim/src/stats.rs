pub struct Stats {
    pub mem: f32,
    pub time: f32,
    pub total_dist: f32,
    pub moves: u32,
    pub hits: u32,
    pub clutter: f32,
    pub coverage: f32
}

impl Stats {
    pub fn dump(&self, label: &str) -> String {
        let mut out = String::new();

        out.push_str("L:");
        out.push_str(label);
        out.push_str("\n");
        out.push_str(&format!("Mem:{}\n", self.mem));
        out.push_str(&format!("Time:{}\n", self.time));
        out.push_str(&format!("Dist:{}\n", self.total_dist));
        out.push_str(&format!("Moves:{}\n", self.moves));
        out.push_str(&format!("Hits:{}\n", self.hits));
        out.push_str(&format!("Clutter:{}\n", self.clutter));
        out.push_str(&format!("Coverage:{}%\n", self.coverage));

        return out;
    }
}