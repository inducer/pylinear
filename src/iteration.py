import os




class IterationSuccessful(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)

class IterationStalled(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)

class IterationStopped(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)

class IterationCountExceeded(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)




class IterationObserver:
    def __init__(self):
        pass

    def reset(self):
        pass

    def add_data_point(self, residual):
        pass




class RelativeGoalDetector(IterationObserver):
    def __init__(self, tolerance):
        IterationObserver.__init__(self)
        self.Tolerance = tolerance

    def reset(self):
        self.FirstResidual = None

    def add_data_point(self, residual):
        if self.FirstResidual is None:
            self.FirstResidual = residual
        else:
            if residual < self.Tolerance * self.FirstResidual:
                raise IterationSuccessful, "Iteration successful."




class IterationStallDetector(IterationObserver):
    def __init__(self, tolerance, max_stalls = 1):
        IterationObserver.__init__(self)
        self.Tolerance = tolerance
        self.MaxStalls = max_stalls

    def reset(self):
        self.BiggestProgress = 1.
        self.LastResidual = None
        self.Stalls = 0

    def add_data_point(self, residual):
        if self.LastResidual is not None:
            progress = max(1, self.LastResidual / residual)
            self.BiggestProgress = max(progress, self.BiggestProgress)
            stalled = False
            if self.BiggestProgress - 1 < 1e-10:
                stalled = True
            else:
                if (progress - 1)/(self.BiggestProgress - 1) < self.Tolerance:
                    stalled = True

            if stalled:
                self.Stalls += 1
                if self.Stalls >= self.MaxStalls:
                    raise IterationStalled, "No progress is being made."
            else:
                self.Stalls = 0
        self.LastResidual = residual




class LastChangeLessThan(IterationObserver):
    def __init__(self, tolerance, max_stalls = 1):
        IterationObserver.__init__(self)
        self.Tolerance = tolerance
        self.MaxStalls = max_stalls

    def reset(self):
        self.LastResidual = None
        self.Stalls = 0

    def add_data_point(self, residual):
        if self.LastResidual is not None:
            stalled = False
            if (self.LastResidual - residual)/self.LastResidual < self.Tolerance:
                # (condition includes the case of LastResidual < residual)
                stalled = True

            if stalled:
                self.Stalls += 1
                if self.Stalls >= self.MaxStalls:
                    raise IterationStalled, "No progress is being made."
            else:
                self.Stalls = 0
        self.LastResidual = residual




class InteractiveStopper(IterationObserver):
    def add_data_point(self, residual):
        filename = "STOP-ITERATION-%d" % os.getpid() 
        try:
            file(filename, "r")
            os.unlink(filename)
            raise IterationStopped, "Stop requested by user."
        except IOError:
            pass




class MaxIterationCountChecker(IterationObserver):
    def __init__(self, max_iterations):
        IterationObserver.__init__(self)
        self.MaxIterations = max_iterations
        self.IterationNumber = 0

    def add_data_point(self, residual):
        self.IterationNumber += 1
        if self.IterationNumber >= self.MaxIterations:
            raise IterationCountExceeded("Maximum number of iterations reached.")
        
  

class CombinedObserver(IterationObserver):
    def __init__(self, observers = []):
        IterationObserver.__init__(self)
        self.Observers = observers[:]

    def reset(self):
        for obs in self.Observers:
            obs.reset()

    def add_data_point(self, residual):
        for obs in self.Observers:
            obs.add_data_point(residual)




def make_observer(max_it = None, 
                 stall_thresh = None, max_stalls = 1, 
                 min_change = None, max_unchanged = 1, 
                 rel_goal = None):
    """Only ever refer to the arguments of this routine by keyword, since
    argument order is not guaranteed to be stable.
    """
    observers = [InteractiveStopper()]

    if max_it is not None:
        observers.append(MaxIterationCountChecker(max_it))

    if stall_thresh is not None:
        observers.append(IterationStallDetector(stall_thresh, max_stalls))

    if rel_goal is not None:
        observers.append(RelativeGoalDetector(rel_goal))

    if min_change is not None:
        observers.append(LastChangeLessThan(min_change, max_unchanged))

    return CombinedObserver(observers)
