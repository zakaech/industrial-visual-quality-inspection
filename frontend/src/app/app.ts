import { Component } from '@angular/core';

import { InspectionPage } from './pages/inspection/inspection-page';

@Component({
  selector: 'app-root',
  imports: [InspectionPage],
  template: '<app-inspection-page />',
})
export class App {}
