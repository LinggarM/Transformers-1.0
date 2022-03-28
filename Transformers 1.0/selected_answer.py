class SelectedAnswer :

  def __init__ (self, idx_sorted, percentage, njawaban) :

    # idx_sorted
    self.idx_sorted = idx_sorted

    # set total selected answers
    self.nSelected(percentage, njawaban)

    # set total selected answers for each part (top, middle, bottom)
    self.nSelectedEach()

    # set rest of total selected answers
    self.nSelectedRest()

    # set selected answers index
    self.setIdxSelected(njawaban)

  def nSelected(self, percentage, njawaban) :
    self.nselected = int(percentage/100 * njawaban)
    if (self.nselected < 3) :
      self.nselected = 3
  
  def nSelectedEach(self) :
    self.nselected_each = int(self.nselected / 3)
  
  def nSelectedRest(self) :
    self.nselected_rest = int(self.nselected % 3)
  
  def setIdxSelected(self, njawaban) :
    nextra_top, nextra_bottom = self.getExtra()
    idx_top = self.getIdxTop(nextra_top)
    idx_middle = self.getIdxMiddle(njawaban)
    idx_bottom = self.getIdxBottom(nextra_bottom)
    self.idx_selected = idx_top + idx_middle + idx_bottom
  
  def getExtra(self) :
    nextra_top = 0
    nextra_bottom = 0
    if (self.nselected_rest > 0) :
      nextra_top += 1
      if (self.nselected_rest > 1) :
        nextra_bottom += 1
    return nextra_top, nextra_bottom
  
  def getIdxTop(self, nextra_top) :
    idx_top = []
    for i in range(self.nselected_each + nextra_top) :
      idx_top.append(self.idx_sorted[i])
    return idx_top
  
  def getIdxMiddle(self, njawaban) :
    start_idx = int((njawaban/2) - (self.nselected_each/2))
    idx_middle = []
    for i in range(self.nselected_each) :
      idx_middle.append(self.idx_sorted[start_idx:][i])
    return idx_middle

  def getIdxBottom(self, nextra_bottom) :
    idx_bottom = []
    for i in range(self.nselected_each + nextra_bottom) :
      idx_bottom.append(np.flip(self.idx_sorted)[i])
    return idx_bottom
  
  def setLabels(self, labels) :
    self.labels = [int(i) for i in labels]