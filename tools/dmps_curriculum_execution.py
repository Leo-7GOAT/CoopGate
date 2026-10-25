"""Training-only SUMO reset adapter; actual cost and execution are unchanged."""
from coopgate.dmps_curriculum_initials import initialize_native,phase_at
from train_dmps_observable_development import ObservableSumoExecution,io


class CurriculumExecution(ObservableSumoExecution):
    def reset(self):
        super().reset()
        phase=self.case.get('phase',phase_at(len(self.calls)))
        try:
            evidence=initialize_native(self.raw,self.case['seed'],phase)
            io.exclusive_json(self.directory/'initialization.json',dict(**evidence,
                starting_actual_decisions=len(self.calls)))
            return self.obs()
        except Exception as error:
            io.exclusive_json(self.directory/'curriculum_reset_failure.json',dict(case=self.case,
                type=type(error).__name__,message=str(error)))
            self.recorded=True
            raise
