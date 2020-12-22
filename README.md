<p align="center">
    <img src="https://user-images.githubusercontent.com/39633205/97300670-97643100-1878-11eb-90e3-c1d979111cda.png" alt="Logo"/>
</p>

# Scout Rover
### Autonomous Exploration and Mapping of Unknown Environments

## About
Using only a single depth camera, **Scout Rover** can move around autonomously to explore its surroundings and generate a 2D map from its observations.

We have created a system which can explore an unknown area and generate its 2D map autonomously. It gets information about its surrounding from a **single forward-facing camera** which is used to estimate motion using **Visual Odomtery**. This raw information is filtered by a long pipeline. It is first filtered to check whether the frames are all equal i.e. there is no motion between corresponding frames. If, yes, the frame is not processed to avoid unneeded computations. These filtered frames are then checked for motion blur which can easily slip into the raw data because of the motion of the rover. Blurred frames are sharpened to improve their pixel information. Finally, these frames go into the feature detection and matching process to compute relative motion.

To cater the noise and perform better localization of the rover, we have implemented **Monte Carlo Localization** which helps to get better estimate of the *true location* of the rover. Otherwise, it might bump into obstacles due to poor path planning.

## Summary of the technologies / methodoligies
- **Visual Sensor**: Microsoft Kinect v1
- **Features Detected**: ORB (Oriented FAST and Rotated Brief)
- **Matching Algorithm**: FLANN Based Matcher
- **Localization**: Monte Carlo Localization

## Contribution Guidelines
Each commit must have only atomic changes i.e. not more than 1 update. For example, it should not be the case that a single commit resolves an error and also implements a new function/feature.

The commit messages should follow the following pattern:
```
[<TYPE>][<PLATFORM>] <DESCRIPTION>
    |       |
    |       |
    |       ------> [Server] | [Rover] | [Both]
    |
    ------> [Bug Fix] | [Feature] | [Refactor] | [Docs] 
```

**TYPE:** Explain the general purpose
  - Bug Fix
  - Feature
  - Refactor
  - Docs

**PLATFORM:** Whether the change is on the server side or rover side or both
  - Arduino
  - Server
  - Rover
  - Both

**DESCRIPTION:** Explain the specific changes

**Examples**
```
[Bug Fix][Server] Resolve INDEX OUT OF RANGE error in keypoints matching
[Feature][Rover] Detect features in input video frame
[Refactor][Both] Code structured in NetworkMessage.py
[Docs][Server] Add docstring for myFunc()
```
