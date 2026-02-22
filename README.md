# smart-racing-data# 
Electrathon Ghost Line

A driver profiling and track optimization system for IEEE Electrathon vehicles.  
This project helps drivers analyze previous races and compare their performance against optimal trajectories.

## Features

- Data acquisition from GPS, IMU, and CANBUS telemetry.
- Track mapping using the "slow lap" method.
- Vehicle dynamics simulation and parameter modeling.
- Lap-time and energy-optimal trajectory generation.
- Real-time feedback for drivers with "ghost line" display.
- Webpage interface to view driver statistics, past races, and improvement tips.

## Project Roadmap

1. **Data Acquisition**
   - Collect GPS and IMU data along the track.
   - Calculate track width and reference points.
2. **Vehicle Modeling**
   - Convert SolidWorks parameters into simulation-ready input.
   - Include mass, motor curves, drag, and tire friction.
3. **Trajectory Optimization**
   - Generate energy-limited optimal racing lines.
   - Output trajectory data for analysis and feedback.
4. **Ghost Line Feedback**
   - Compare driver position and speed to optimal trajectory.
   - Display real-time guidance on steering, throttle, and braking.
5. **Web Interface**
   - Display individual driver statistics.
   - Provide improvement recommendations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow the structure above and document new data or features clearly.
