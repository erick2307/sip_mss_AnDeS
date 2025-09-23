# User-Controlled Years Back Feature

## 🎯 **Feature Overview**

Added user control to specify how many years back from the event date to include in the monthly time series analysis, replacing the default "2016 to event year" behavior.

---

## ✨ **New Functionality**

### **User Interface Addition:**
- **Radio Button Selection:** Choose between "All available years (2016 onwards)" or "Custom years back"
- **Slider Control:** When custom is selected, users can specify 1 to maximum available years
- **Smart Limits:** Automatically calculated based on available data (limited by 2016 as earliest year)
- **Real-time Info:** Analysis period displayed in sidebar shows exact year range

### **Backend Logic Enhancement:**
- **Modified Function:** `load_mss_data_monthly_timeseries()` now accepts `years_back` parameter
- **Smart Validation:** Ensures start year never goes before 2016 (data availability limit)
- **Flexible Range:** `start_year = max(2016, end_year - years_back + 1)`
- **Backward Compatibility:** When `years_back=None`, defaults to original behavior (2016 onwards)

---

## 🔧 **Technical Implementation**

### **Function Signature Update:**
```python
def load_mss_data_monthly_timeseries(event_datetime: datetime, 
                                    mesh_id_list: List[str], 
                                    multi_mesh_analysis: bool = False,
                                    years_back: int = None) -> pd.DataFrame:
```

### **Logic Changes:**
```python
if years_back is None:
    # Default behavior: from 2016 to event year
    start_year = 2016
else:
    # User-specified years back
    start_year = max(2016, end_year - years_back + 1)  # Ensure we don't go before 2016
```

### **Configuration Storage:**
```python
st.session_state.config = {
    # ... existing parameters ...
    'years_back': years_back,  # New parameter stored
    # ... other parameters ...
}
```

---

## 📊 **User Experience**

### **Default Behavior:**
- **Radio Selection:** "All available years (2016 onwards)" selected by default
- **Analysis Range:** All years from 2016 to event year (original behavior)
- **No Change:** Existing users see familiar behavior

### **Custom Selection:**
- **Radio Selection:** User switches to "Custom years back"
- **Slider Appears:** Dynamic slider with appropriate min/max values
- **Real-time Updates:** Analysis info updates immediately
- **Smart Defaults:** Slider defaults to 5 years or maximum available

### **Example Scenarios:**

#### **Event in 2024, Custom 3 Years Back:**
- **Selection:** Custom years back = 3
- **Analysis Period:** 2022, 2023, 2024 (3 years)
- **Display:** "1-month periods from 3 years (2022-2024)"

#### **Event in 2018, Custom 5 Years Back:**
- **Selection:** Custom years back = 5
- **Analysis Period:** 2016, 2017, 2018 (only 3 years available)
- **Smart Limit:** Slider max = 3, automatically limited by 2016 boundary
- **Display:** "1-month periods from 3 years (2016-2018)"

#### **Event in 2025, All Available:**
- **Selection:** All available years
- **Analysis Period:** 2016-2025 (10 years)
- **Display:** "1-month periods from 10 years (2016-2025)"

---

## 🎨 **UI Components**

### **Sidebar Section:**
```
📅 Data Period Selection
○ All available years (2016 onwards)
○ Custom years back

[When Custom selected:]
Years back from event year: [1] ←→ [9]
```

### **Analysis Info Display:**
```
📅 Event DateTime: 2024-01-01 16:00
📊 Analysis: 1-month periods from 3 years (2022-2024)
🔍 Monthly Period: 1 month before until event date/time each year
```

---

## 💡 **Benefits**

### **For Users:**
- **Focused Analysis:** Can limit to recent years for more relevant patterns
- **Performance Control:** Smaller datasets process faster
- **Comparative Studies:** Easy to compare "last 3 years" vs "last 5 years"
- **Event-Specific:** Match analysis period to event context

### **For Analysis:**
- **Pattern Relevance:** Focus on recent patterns that may be more relevant
- **Processing Speed:** Smaller datasets = faster matrix profile computation
- **Memory Efficiency:** Reduced memory usage for large historical ranges
- **Statistical Focus:** More focused statistical thresholds

### **For Research:**
- **Methodology Control:** Precise control over analysis window
- **Comparative Studies:** Easy to test different historical depths
- **Event Studies:** Match analysis period to event characteristics
- **Publication Standards:** Clear documentation of analysis period

---

## 🔍 **Data Validation & Safety**

### **Boundary Checks:**
- ✅ **Lower Bound:** `start_year = max(2016, calculated_start)` prevents going before 2016
- ✅ **Upper Bound:** Slider max automatically calculated as `event_year - 2016 + 1`
- ✅ **Logic Validation:** `start_year <= end_year` always maintained
- ✅ **Edge Cases:** Single year events (years_back=1) properly handled

### **Error Handling:**
- **Invalid Ranges:** Logged with clear error messages
- **Missing Data:** Graceful handling when specific years unavailable
- **User Feedback:** Real-time validation and clear error messages
- **Fallback:** Defaults to safe behavior when parameters invalid

---

## 📈 **Performance Impact**

### **Positive Impacts:**
- **Faster Loading:** Fewer years = faster data loading
- **Reduced Memory:** Smaller datasets require less memory
- **Quicker Detection:** Matrix profile computation scales with data size
- **Better Caching:** Streamlit cache more effective with smaller datasets

### **Maintained Quality:**
- **Algorithm Unchanged:** Matrix profile logic unchanged
- **Pattern Detection:** Still effective with focused datasets
- **Statistical Validity:** Threshold methods adapted appropriately
- **Visual Quality:** All visualizations work with any year range

---

## 🧪 **Testing Scenarios**

### **Basic Functionality:**
- ✅ Default selection (all years) works as before
- ✅ Custom selection with various year counts
- ✅ Boundary cases (1 year, maximum years)
- ✅ Configuration storage and retrieval

### **Edge Cases:**
- ✅ Events in 2016 (minimum year)
- ✅ Events in 2025 (maximum current year)
- ✅ February 29 events (leap year handling)
- ✅ January events requiring previous year data

### **User Experience:**
- ✅ Immediate UI feedback on selection changes
- ✅ Clear analysis period display
- ✅ Intuitive slider behavior
- ✅ Helpful tooltips and guidance

---

## 🚀 **Next Steps**

1. **User Testing:** Validate with real-world use cases
2. **Performance Monitoring:** Track loading times with different year ranges
3. **User Feedback:** Gather input on default values and UI placement
4. **Documentation Update:** Update user guides with new feature

This enhancement provides users with much better control over their analysis while maintaining all existing functionality! 🎉