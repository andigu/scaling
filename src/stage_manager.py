#!/usr/bin/env python3
"""
Stage Manager for 3-stage curriculum learning.

Coordinates the progression through:
1. Stage 1: Train at low noise (p~0.5) until convergence
2. Stage 2: Curriculum learning - gradually increase p  
3. Stage 3: Train at high noise for final convergence
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for 3-stage curriculum learning."""
    enabled: bool = False
    stage1_p: float = 0.5
    stage1_steps: int = 25000
    stage2_p_end: float = 2.1
    stage2_steps: int = 25000
    stage3_steps: int = 25000
    
    def __post_init__(self):
        """Validate configuration."""
        if self.enabled:
            assert self.stage1_steps > 0, "stage1_steps must be positive"
            assert self.stage2_steps > 0, "stage2_steps must be positive"
            assert self.stage3_steps > 0, "stage3_steps must be positive"
            assert self.stage1_p < self.stage2_p_end, "stage2_p_end must be greater than stage1_p"
    
    @property
    def stage2_p_start(self) -> float:
        """Stage 2 starting p is always equal to stage1_p for smooth transition."""
        return self.stage1_p
    
    @property
    def total_steps(self) -> int:
        """Total steps across all stages."""
        return self.stage1_steps + self.stage2_steps + self.stage3_steps
    
    @property
    def stage1_end_step(self) -> int:
        """Global step where stage 1 ends."""
        return self.stage1_steps
    
    @property
    def stage2_end_step(self) -> int:
        """Global step where stage 2 ends."""
        return self.stage1_steps + self.stage2_steps
    
    @property
    def stage3_end_step(self) -> int:
        """Global step where stage 3 ends."""
        return self.total_steps


class StageManager:
    """Manages curriculum learning stages and transitions."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self._current_stage = 1
        self._stage_transitions_logged = set()
        
        if config.enabled:
            log.info(f"Initialized 3-stage curriculum learning:")
            log.info(f"  Stage 1: p={config.stage1_p:.1f} for {config.stage1_steps} steps")
            log.info(f"  Stage 2: p={config.stage2_p_start:.1f}→{config.stage2_p_end:.1f} over {config.stage2_steps} steps")
            log.info(f"  Stage 3: p={config.stage2_p_end:.1f} for {config.stage3_steps} steps")
            log.info(f"  Total steps: {config.total_steps}")
    
    def get_current_stage(self, global_step: int) -> int:
        """Determine current stage based on global step."""
        if not self.config.enabled:
            return 1  # Default stage for non-curriculum training
        
        if global_step < self.config.stage1_end_step:
            return 1
        elif global_step < self.config.stage2_end_step:
            return 2
        else:
            return 3
    
    def get_current_p(self, global_step: int) -> float:
        """Calculate current p value based on stage and step."""
        if not self.config.enabled:
            raise RuntimeError("Curriculum learning is not enabled, get_current_p should not be called.")
        
        stage = self.get_current_stage(global_step)
        
        if stage == 1:
            return self.config.stage1_p
        
        elif stage == 2:
            # Linear curriculum progression
            stage2_local_step = global_step - self.config.stage1_end_step
            progress = stage2_local_step / self.config.stage2_steps
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            return self.config.stage2_p_start + \
                   (self.config.stage2_p_end - self.config.stage2_p_start) * progress
        
        else:  # stage == 3
            return self.config.stage2_p_end
    
    def get_stage_prefix(self, global_step: int) -> str:
        """Get logging prefix for current stage."""
        if not self.config.enabled:
            return ""  # No prefix for non-curriculum training
        
        stage = self.get_current_stage(global_step)
        return f"stage{stage}/"
    
    def check_stage_transition(self, global_step: int) -> Optional[Tuple[int, int]]:
        """Check if a stage transition occurred. Returns (old_stage, new_stage) or None."""
        current_stage = self.get_current_stage(global_step)
        
        if current_stage != self._current_stage:
            old_stage = self._current_stage
            self._current_stage = current_stage
            
            # Log transition only once
            if (old_stage, current_stage) not in self._stage_transitions_logged:
                self._log_stage_transition(old_stage, current_stage, global_step)
                self._stage_transitions_logged.add((old_stage, current_stage))
            
            return (old_stage, current_stage)
        
        return None
    
    def _log_stage_transition(self, old_stage: int, new_stage: int, global_step: int):
        """Log stage transition."""
        current_p = self.get_current_p(global_step)
        log.info(f"=== STAGE TRANSITION: Stage {old_stage} → Stage {new_stage} ===")
        log.info(f"Global step: {global_step}")
        log.info(f"Current p: {current_p:.3f}")
        
        if new_stage == 2:
            log.info(f"Starting curriculum learning: p={self.config.stage2_p_start:.1f}→{self.config.stage2_p_end:.1f}")
        elif new_stage == 3:
            log.info(f"Starting final stage at p={self.config.stage2_p_end:.1f}")
    
    def get_stage_progress(self, global_step: int) -> dict:
        """Get detailed progress information for current stage."""
        stage = self.get_current_stage(global_step)
        
        if stage == 1:
            stage_local_step = global_step
            stage_total_steps = self.config.stage1_steps
            stage_progress = stage_local_step / stage_total_steps
        elif stage == 2:
            stage_local_step = global_step - self.config.stage1_end_step
            stage_total_steps = self.config.stage2_steps
            stage_progress = stage_local_step / stage_total_steps
        else:  # stage == 3
            stage_local_step = global_step - self.config.stage2_end_step
            stage_total_steps = self.config.stage3_steps
            stage_progress = stage_local_step / stage_total_steps
        
        return {
            'current_stage': stage,
            'stage_local_step': stage_local_step,
            'stage_total_steps': stage_total_steps,
            'stage_progress': min(stage_progress, 1.0),
            'global_step': global_step,
            'total_steps': self.config.total_steps,
            'global_progress': global_step / self.config.total_steps,
            'current_p': self.get_current_p(global_step)
        }
    
    def is_training_complete(self, global_step: int) -> bool:
        """Check if all stages are complete."""
        if not self.config.enabled:
            return False  # Let normal max_steps handle this
        
        return global_step >= self.config.total_steps
    
    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            'current_stage': self._current_stage,
            'stage_transitions_logged': list(self._stage_transitions_logged)
        }
    
    def load_state_dict(self, state_dict: dict):
        """Restore state from checkpoint."""
        self._current_stage = state_dict.get('current_stage', 1)
        self._stage_transitions_logged = set(state_dict.get('stage_transitions_logged', []))