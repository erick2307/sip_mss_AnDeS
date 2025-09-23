# AnDeS Presentation Structure

## Presentation Overview
**Title**: AnDeS - ANomaly DEtection System for Mobile Spatial Statistics
**Duration**: 20-30 minutes  
**Audience**: Technical stakeholders, researchers, disaster management professionals
**Objective**: Demonstrate capabilities and practical applications of the AnDeS system

---

## Slide Deck Structure

### **Section 1: Introduction & Context (3-4 slides)**

#### Slide 1: Title Slide
- **Title**: AnDeS - ANomaly DEtection System
- **Subtitle**: Real-time Anomaly Detection in Mobile Spatial Statistics Data
- **Author**: Erick Mas, Tohoku University
- **Version**: 2025.1.0
- **Date**: [Presentation Date]

#### Slide 2: Problem Statement
- **Challenge**: Detecting unusual population movement patterns during emergencies
- **Context**: Japan's need for real-time disaster response monitoring
- **Data Source**: Mobile Spatial Statistics (MSS) from mobile network operators
- **Goal**: Early detection of anomalous events through population movement analysis

#### Slide 3: Solution Overview
- **System Name**: AnDeS (ANomaly DEtection System)
- **Core Technology**: Matrix Profile-based anomaly detection
- **Platform**: Web-based Streamlit application
- **Data Coverage**: 2016-2025, hourly resolution, nationwide mesh codes
- **Key Innovation**: Event-focused historical comparison methodology

---

### **Section 2: Technical Foundation (4-5 slides)**

#### Slide 4: Data Architecture
- **Input Data Types**:
  - MSS Population Data (`ntt_mss_{year}.npy`)
  - Geographic Mesh Mapping (`ntt_mss_{year}_areas.npy`)
- **Data Characteristics**:
  - Temporal: Hourly measurements (8,760 points/year)
  - Spatial: Japanese mesh code system
  - Coverage: National scale, 10+ years historical
- **Data Flow Diagram**: Raw Data → Processing → Analysis → Visualization

#### Slide 5: Matrix Profile Algorithm
- **Core Concept**: Time series subsequence similarity analysis
- **Process Flow**:
  1. Sliding window pattern extraction
  2. Distance calculation between all subsequences
  3. Nearest neighbor identification
  4. Anomaly threshold application
- **Implementation Options**: PySCAMP (GPU), STUMPY (CPU), Custom fallback
- **Key Parameters**: Subsequence length, threshold multiplier, normalization

#### Slide 6: Monthly Time Series Innovation
- **Traditional Approach**: Continuous date range analysis
- **AnDeS Innovation**: Event-focused monthly period extraction
- **Methodology**:
  - Extract 1-month periods from multiple years
  - Same calendar dates across years for seasonal consistency
  - Concatenate periods for enhanced baseline comparison
- **Benefits**: Better historical context, seasonal relevance, improved detection accuracy

#### Slide 7: System Architecture
- **Frontend**: Interactive Streamlit web interface
- **Core Engine**: Matrix profile anomaly detection
- **Data Management**: Lazy loading with intelligent caching
- **Monitoring**: Real-time system performance tracking
- **Memory Management**: Efficient handling of large datasets (2-4GB typical)

---

### **Section 3: User Interface & Features (3-4 slides)**

#### Slide 8: Interface Overview
- **Four Main Tabs**:
  - 📊 Real-time Analysis: Detection execution and results
  - 📈 Historical View: Statistical analysis and patterns
  - ⚙️ System Status: Performance monitoring
  - 📚 Documentation: Usage guides and specifications
- **Key Design Principles**: Intuitive navigation, real-time feedback, mobile-friendly

#### Slide 9: Configuration & Parameters
- **Event Selection**: Predefined major events vs. custom analysis
- **Detection Parameters**:
  - Subsequence Length: 6-168 hours (pattern window)
  - Threshold Multiplier: 1.0-5.0 (sensitivity control)
  - Geographic Scope: Single mesh vs. multi-mesh analysis
- **Data Options**: Year range, mesh ID selection, area aggregation

#### Slide 10: Visualization Capabilities
- **Time Series Plots**: Population trends with anomaly overlays
- **Anomaly Heatmaps**: Binary detection grid (day × hour)
- **Score Heatmaps**: Continuous anomaly intensity visualization
- **Statistical Charts**: Distribution analysis and correlation studies
- **Interactive Features**: Zoom, pan, hover details, downloadable results

---

### **Section 4: Practical Applications (4-5 slides)**

#### Slide 11: Predefined Event Analysis
- **Natural Disasters**:
  - 2024 Noto Peninsula Earthquake (Mw 7.5)
  - 2016 Kumamoto Earthquakes (Mw 7.0)
  - 2018 Japan Floods (Hiroshima/Okayama)
- **Transportation Incidents**:
  - 2024 Haneda Airport Runway Collision
- **Large Gatherings**:
  - Taylor Swift Tokyo Dome Concert
  - Comic Market 104, Tokyo Game Show

#### Slide 12: Case Study Example
- **Event**: [Select specific event, e.g., Noto Peninsula Earthquake]
- **Analysis Setup**:
  - Geographic focus: 3×3 mesh grid around epicenter
  - Time period: December 2023 - January 2024
  - Historical comparison: Same periods 2016-2023
- **Results Preview**: Show sample visualization
- **Key Findings**: Detected anomalies and their significance

#### Slide 13: Detection Results Interpretation
- **Anomaly Indicators**:
  - Population spikes (evacuation gathering points)
  - Population drops (evacuation zones)
  - Unusual temporal patterns (night-time activity)
- **Threshold Tuning**: δ=3.0 for conservative detection, δ=2.0 for sensitive detection
- **Validation Methods**: Cross-reference with known event timelines

#### Slide 14: Real-world Applications
- **Emergency Response**: Early warning system integration
- **Urban Planning**: Understanding population flow patterns
- **Research Applications**: Disaster response behavior analysis
- **Policy Support**: Evidence-based emergency planning
- **Infrastructure Management**: Resource allocation optimization

---

### **Section 5: Technical Performance (2-3 slides)**

#### Slide 15: System Performance Metrics
- **Processing Speed**:
  - Monthly analysis: 2-5 minutes typical
  - Multi-year analysis: 15-30 minutes
  - Real-time monitoring: Sub-second updates
- **Memory Management**:
  - Typical usage: 2-4 GB RAM
  - Cache optimization: 2-year rolling window
  - Lazy loading: On-demand data access
- **Accuracy Metrics**: [Include specific validation results if available]

#### Slide 16: Scalability & Compatibility
- **Data Scale**: Tested with 10+ years of national data
- **Geographic Coverage**: Full Japan mesh code support
- **Platform Compatibility**: Cross-platform Python deployment
- **Library Support**: NumPy 2.0 compatibility, multiple matrix profile backends
- **Hardware Requirements**: Standard desktop/laptop sufficient

---

### **Section 6: Advanced Features (2-3 slides)**

#### Slide 17: Quality Assurance & Reliability
- **Data Validation**: Automatic range checking and completeness verification
- **Error Handling**: Graceful degradation with fallback options
- **Monitoring**: Real-time system health and performance tracking
- **Logging**: Comprehensive session activity recording
- **Testing**: Automated unit and integration test suites

#### Slide 18: Innovation Highlights
- **NumPy 2.0 Compatibility**: Automatic patches for library compatibility
- **Adaptive Algorithm Selection**: Intelligent fallback system
- **Memory Optimization**: Efficient large dataset handling
- **Event-Focused Analysis**: Novel monthly extraction methodology
- **Real-time Monitoring**: Live system status reporting

---

### **Section 7: Future Directions (2 slides)**

#### Slide 19: Enhancement Roadmap
- **Algorithm Improvements**:
  - Streaming real-time detection
  - Multi-scale temporal analysis
  - Ensemble detection methods
- **Data Integration**:
  - External data source incorporation (weather, social media)
  - Real-time MSS feed connection
  - Multi-modal analysis capabilities
- **Interface Enhancements**:
  - Mobile optimization
  - 3D visualization
  - Collaborative analysis features

#### Slide 20: Research Opportunities
- **Methodological Research**:
  - Improved threshold determination methods
  - Spatial-temporal correlation analysis
  - Machine learning integration
- **Application Research**:
  - Cross-disaster type comparison
  - Long-term trend analysis
  - International applicability studies
- **Technical Research**:
  - Real-time processing optimization
  - Distributed computing implementation
  - Edge computing deployment

---

### **Section 8: Conclusion (2 slides)**

#### Slide 21: Key Achievements
- **Technical**: Robust matrix profile implementation with multiple backends
- **Methodological**: Novel monthly time series analysis approach
- **Practical**: User-friendly interface for complex analysis
- **Scalable**: Efficient handling of large-scale spatiotemporal data
- **Reliable**: Comprehensive error handling and quality assurance

#### Slide 22: Summary & Contact
- **AnDeS Value Proposition**: 
  - Real-time anomaly detection in population movement
  - Historical context for improved accuracy
  - User-friendly interface for complex analysis
  - Proven application to major Japanese events
- **Contact Information**: [Your contact details]
- **Repository**: GitHub link for source code access
- **Questions & Discussion**

---

## Presentation Notes & Tips

### **Technical Demonstration Section (Optional 5-10 minutes)**
If doing a live demo, focus on:
1. **Quick Setup**: Show the interface loading
2. **Event Selection**: Choose a well-known event (e.g., Noto Peninsula Earthquake)
3. **Parameter Configuration**: Demonstrate sensitivity adjustment
4. **Results Visualization**: Show time series, heatmaps, and details
5. **System Status**: Highlight real-time monitoring capabilities

### **Audience-Specific Adaptations**

#### For Technical Audiences:
- Emphasize algorithm details and implementation choices
- Include more technical performance metrics
- Discuss code architecture and design patterns
- Show advanced configuration options

#### For Management/Policy Audiences:
- Focus on practical applications and benefits
- Emphasize cost-effectiveness and scalability
- Highlight decision-support capabilities
- Include ROI considerations

#### For Research Audiences:
- Emphasize methodological innovations
- Include validation studies and accuracy metrics
- Discuss research opportunities and limitations
- Compare with existing approaches

### **Visual Design Recommendations**

#### Slide Design:
- **Color Scheme**: Use consistent blue/green palette matching the app
- **Typography**: Clean, professional fonts (Arial/Helvetica)
- **Layout**: Maximum 6-7 bullet points per slide
- **Diagrams**: Use flowcharts for data flow and process explanation

#### Data Visualizations:
- **High-Resolution**: Ensure plots are clear when projected
- **Annotations**: Label key features and anomalies clearly
- **Consistency**: Use same color coding across all visualizations
- **Context**: Include titles, axes labels, and legends

### **Backup Slides (Optional)**
Prepare additional slides for detailed Q&A:
- Detailed algorithm mathematics
- Complete system requirements
- Full event catalog
- Extended performance metrics
- Troubleshooting examples

### **Handout Materials**
Consider providing:
- System requirements checklist
- Quick start guide
- Event analysis examples
- Contact information and resources

This presentation structure provides a comprehensive overview of the AnDeS system while maintaining flexibility for different audiences and time constraints.