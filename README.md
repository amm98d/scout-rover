![logo-no-background](https://user-images.githubusercontent.com/39633205/97300670-97643100-1878-11eb-90e3-c1d979111cda.png)
# Scout Rover
### Autonomous Exploration and Mapping of Unknown Environments

Using only a single depth camera, **Scout Rover** can move around autonomously to explore its surroundings and generate a 2D map from its observations.

## Contributing Guidelines
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
