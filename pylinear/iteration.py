class tIterationSuccessful(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)

class tIterationStalled(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)

class tIterationCountExceeded(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)




class tIterationObserver:
    def __init__(self):
        pass

    def reset(self):
        pass

    def addDataPoint(self, residual):
        pass




class tRelativeGoalDetector(tIterationObserver):
    def __init__(self, tolerance):
        tIterationObserver.__init__(self)
        self.Tolerance = tolerance

    def reset(self):
        self.FirstResidual = None

    def addDataPoint(self, residual):
        if self.FirstResidual is None:
            self.FirstResidual = residual
        else:
            if residual < self.Tolerance * self.FirstResidual:
                raise tIterationSuccessful, "Iteration successful."




class tIterationStallDetector(tIterationObserver):
    def __init__(self, tolerance, max_stalls = 1):
        tIterationObserver.__init__(self)
        self.Tolerance = tolerance
        self.MaxStalls = max_stalls

    def reset(self):
        self.BiggestProgress = 1.
        self.LastResidual = None
        self.Stalls = 0

    def addDataPoint(self, residual):
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
                    raise tIterationStalled, "No progress is being made."
            else:
                self.Stalls = 0
        self.LastResidual = residual




class tMaxIterationCountChecker(tIterationObserver):
    def __init__(self, max_iterations):
        tIterationObserver.__init__(self)
        self.MaxIterations = max_iterations
        self.IterationNumber = 0

    def addDataPoint(self, residual):
        self.IterationNumber += 1
        if self.IterationNumber >= self.MaxIterations:
            raise tIterationCountExceeded("Maximum number of iterations reached.")
        
  

class tCombinedObserver(tIterationObserver):
    def __init__(self, observers = []):
        tIterationObserver.__init__(self)
        self.Observers = observers[:]

    def reset(self):
        for obs in self.Observers:
            obs.reset()

    def addDataPoint(self, residual):
        for obs in self.Observers:
            obs.addDataPoint(residual)




def makeObserver(max_it = None, stall_thresh = None, max_stalls = 1, rel_goal = None):
    """Only ever refer to the arguments of this routine by keyword, since
    argument order is not guaranteed to be stable.
    """
    observers = []
    if max_it is not None:
        observers.append(tMaxIterationCountChecker(max_it))

    if stall_thresh is not None:
        observers.append(tIterationStallDetector(stall_thresh, max_stalls))

    if rel_goal is not None:
        observers.append(tRelativeGoalDetector(rel_goal))

    return tCombinedObserver(observers)
