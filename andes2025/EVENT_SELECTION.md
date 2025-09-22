# Event-Based Mesh Selection

## Overview

The AnDeS system now includes predefined major events with pre-configured mesh codes for easier analysis of specific incidents and occasions.

## Available Events

### 1. Haneda Airport Runway Collision
- **Date**: January 2, 2024, 17:00 UTC
- **Main Mesh Code**: 533937621
- **Total Mesh Codes**: 9 surrounding areas
- **Location**: Tokyo Haneda Airport area
- **Description**: Analysis area around the runway collision incident

### 2. Taylor Swift – The Eras Tour (Tokyo Dome)
- **Date**: February 7, 2024, 00:00 UTC
- **Main Mesh Code**: 533946403
- **Total Mesh Codes**: 9 surrounding areas
- **Location**: Tokyo Dome area
- **Description**: Concert event analysis area

### 3. Bruno Mars Concert (Tokyo Dome)
- **Date**: January 1, 2024, 00:00 UTC
- **Main Mesh Code**: 533946403
- **Total Mesh Codes**: 9 surrounding areas
- **Location**: Tokyo Dome area
- **Description**: Concert event analysis area

### 4. Comic Market 104 (Tokyo Big Sight)
- **Date**: August 11, 2024, 10:00 UTC
- **Main Mesh Code**: 533947534
- **Total Mesh Codes**: 9 surrounding areas
- **Location**: Tokyo Big Sight
- **Description**: Major anime/manga convention event

### 5. Tokyo Game Show 2024 (Makuhari Messe)
- **Date**: September 28, 2024, 10:00 UTC
- **Main Mesh Code**: 534041724
- **Total Mesh Codes**: 9 surrounding areas
- **Location**: Makuhari Messe
- **Description**: Gaming industry exhibition event

## How to Use

### Event Selection
1. In the sidebar, go to "Event Selection"
2. Choose from the dropdown:
   - Select "Custom Selection" for manual mesh ID input
   - Select any predefined event for automatic mesh configuration

### Event Options
When an event is selected:
- **Event information** is automatically displayed (date, location, mesh counts)
- **Use all event meshes**: Toggle to use all 9 mesh codes or just the main one
- **Automatic date context**: Event date is shown for reference

### Custom Selection
When "Custom Selection" is chosen:
- **Single mesh selection**: Choose from available mesh IDs
- **Custom mesh input**: Enter multiple mesh IDs separated by commas
- **Multi-mesh analysis**: Option to aggregate data from multiple meshes

## Benefits

1. **Pre-configured Analysis**: No need to manually find mesh codes for major events
2. **Comprehensive Coverage**: Each event includes 9 mesh codes in a 3x3 grid around the main location
3. **Easy Comparison**: Switch between events quickly for comparative analysis
4. **Event Context**: Date and location information provided automatically
5. **Flexible Usage**: Option to use all meshes or just the main mesh for each event

## Technical Implementation

- Events are defined in the `events` list with structured data
- Mesh codes are organized in a 3x3 grid pattern around the main location
- Event dates use UTC timezone for consistency
- Automatic validation and error handling for event selection

## Future Extensions

The event system can be easily extended by adding new events to the `events` list with:
- Event datetime
- Main mesh code
- List of surrounding mesh codes
- Event description
- Configuration options