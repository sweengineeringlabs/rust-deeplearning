//! Application state

use serde::{Deserialize, Serialize};

/// User profile and progress
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct User {
    pub name: String,
    pub avatar: Option<String>,
    pub level: u32,
    pub xp: u32,
    pub xp_to_next_level: u32,
    pub streak: u32,
    pub hearts: u32,
    pub gems: u32,
    pub completed_lessons: Vec<String>,
    pub achievements: Vec<Achievement>,
}

impl Default for User {
    fn default() -> Self {
        Self {
            name: "Learner".to_string(),
            avatar: None,
            level: 1,
            xp: 0,
            xp_to_next_level: 100,
            streak: 0,
            hearts: 5,
            gems: 50,
            completed_lessons: vec![],
            achievements: vec![],
        }
    }
}

impl User {
    /// Add XP and handle level ups
    pub fn add_xp(&mut self, amount: u32) -> Option<LevelUpEvent> {
        self.xp += amount;

        if self.xp >= self.xp_to_next_level {
            self.level += 1;
            self.xp -= self.xp_to_next_level;
            self.xp_to_next_level = self.calculate_next_level_xp();
            self.gems += 10; // Bonus gems on level up

            return Some(LevelUpEvent {
                new_level: self.level,
                gems_earned: 10,
            });
        }
        None
    }

    fn calculate_next_level_xp(&self) -> u32 {
        100 + (self.level * 25)
    }

    /// Lose a heart (wrong answer)
    pub fn lose_heart(&mut self) -> bool {
        if self.hearts > 0 {
            self.hearts -= 1;
            true
        } else {
            false // No hearts left - game over
        }
    }

    /// Restore hearts
    pub fn restore_hearts(&mut self) {
        self.hearts = 5;
    }

    /// Complete a lesson
    pub fn complete_lesson(&mut self, lesson_id: &str) {
        if !self.completed_lessons.contains(&lesson_id.to_string()) {
            self.completed_lessons.push(lesson_id.to_string());
        }
    }

    /// Check if lesson is completed
    pub fn is_lesson_completed(&self, lesson_id: &str) -> bool {
        self.completed_lessons.contains(&lesson_id.to_string())
    }

    /// Unlock achievement
    pub fn unlock_achievement(&mut self, achievement: Achievement) {
        if !self.achievements.iter().any(|a| a.id == achievement.id) {
            self.achievements.push(achievement);
        }
    }
}

/// Level up event
#[derive(Clone, Debug)]
pub struct LevelUpEvent {
    pub new_level: u32,
    pub gems_earned: u32,
}

/// Achievement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub xp_reward: u32,
}

/// Predefined achievements
impl Achievement {
    pub fn first_lesson() -> Self {
        Self {
            id: "first_lesson".to_string(),
            name: "First Steps".to_string(),
            description: "Complete your first lesson".to_string(),
            icon: "🎯".to_string(),
            xp_reward: 50,
        }
    }

    pub fn perfect_lesson() -> Self {
        Self {
            id: "perfect".to_string(),
            name: "Perfectionist".to_string(),
            description: "Complete a lesson with no mistakes".to_string(),
            icon: "⭐".to_string(),
            xp_reward: 100,
        }
    }

    pub fn streak_7() -> Self {
        Self {
            id: "streak_7".to_string(),
            name: "On Fire".to_string(),
            description: "Maintain a 7-day streak".to_string(),
            icon: "🔥".to_string(),
            xp_reward: 200,
        }
    }

    pub fn level_5() -> Self {
        Self {
            id: "level_5".to_string(),
            name: "Rising Star".to_string(),
            description: "Reach level 5".to_string(),
            icon: "⭐".to_string(),
            xp_reward: 150,
        }
    }

    pub fn level_10() -> Self {
        Self {
            id: "level_10".to_string(),
            name: "Expert".to_string(),
            description: "Reach level 10".to_string(),
            icon: "🏆".to_string(),
            xp_reward: 300,
        }
    }
}
