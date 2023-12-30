/// Define a left and a right region for the application.
/// Each region is divided in vertically stacked rectangles.
use std::marker::PhantomData;
use std::rc::Rc;

use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{block::Position, Block, BorderType, Borders, Padding},
    Frame,
};

// Region Base ---------------------------------------------------------------

pub(crate) struct RegionInfo {
    width_percentage: u16,
}

pub(crate) struct RegionRectInfo {
    index: usize,
    title: &'static str,
    height_percentage: u16,
    pub hotkey: char,
}

pub(crate) trait GetRegionInfo {
    fn get_region_info() -> RegionInfo;
    fn get_rect_info(&self) -> RegionRectInfo;
}

pub(crate) struct Region<I: GetRegionInfo> {
    rects: Option<Rc<[Rect]>>,
    info: RegionInfo,
    _i: PhantomData<I>,
}

impl<T: GetRegionInfo> Region<T> {
    fn new() -> Self {
        Self {
            rects: None,
            info: T::get_region_info(),
            _i: PhantomData,
        }
    }

    pub fn rect(&self, region_section: &T) -> Rect {
        match &self.rects {
            Some(rects) => rects[region_section.get_rect_info().index],
            None => Rect::new(0, 0, 0, 0),
        }
    }

    /// Widget to draw the style of a region
    fn block(&self, region_section: &T, is_focused: bool) -> Block {
        let border_color = if is_focused {
            Color::LightRed
        } else {
            Color::DarkGray
        };
        Block::default()
            .title(format!(
                "{} ({})",
                region_section.get_rect_info().title,
                region_section.get_rect_info().hotkey
            ))
            .title_position(Position::Top)
            .title_alignment(Alignment::Center)
            .borders(Borders::all())
            .border_style(Style::default().fg(border_color))
            .border_type(BorderType::Rounded)
            .padding(Padding {
                left: 10,
                right: 10,
                top: 2,
                bottom: 2,
            })
            .style(Style::default().bg(Color::Black))
    }
}

// Left Region --------------------------------------------------------------

#[derive(PartialEq)]
pub(crate) enum LeftRegion {
    Top,
    Middle,
    Bottom,
}

impl LeftRegion {
    fn variants() -> [LeftRegion; 3] {
        [LeftRegion::Top, LeftRegion::Middle, LeftRegion::Bottom]
    }
}

impl GetRegionInfo for LeftRegion {
    fn get_region_info() -> RegionInfo {
        RegionInfo {
            width_percentage: 25,
        }
    }

    fn get_rect_info(&self) -> RegionRectInfo {
        match self {
            LeftRegion::Top => RegionRectInfo {
                index: 0,
                title: "Backend",
                height_percentage: 30,
                hotkey: 'b',
            },
            LeftRegion::Middle => RegionRectInfo {
                index: 1,
                title: "Benches",
                height_percentage: 60,
                hotkey: 'n',
            },
            LeftRegion::Bottom => RegionRectInfo {
                index: 2,
                title: "Action",
                height_percentage: 10,
                hotkey: 'a',
            },
        }
    }
}

// Right Region --------------------------------------------------------------

#[derive(PartialEq)]
pub(crate) enum RightRegion {
    Top,
    Bottom,
}

impl RightRegion {
    fn variants() -> [RightRegion; 2] {
        [RightRegion::Top, RightRegion::Bottom]
    }
}

impl GetRegionInfo for RightRegion {
    fn get_region_info() -> RegionInfo {
        RegionInfo {
            width_percentage: 100 - LeftRegion::get_region_info().width_percentage,
        }
    }

    fn get_rect_info(&self) -> RegionRectInfo {
        match self {
            RightRegion::Top => RegionRectInfo {
                index: 0,
                title: "Results",
                height_percentage: 90,
                hotkey: 'r',
            },
            RightRegion::Bottom => RegionRectInfo {
                index: 1,
                title: "Progress",
                height_percentage: 10,
                hotkey: 'p',
            },
        }
    }
}

// Regions definition --------------------------------------------------------

pub enum FocusedRegion<L: GetRegionInfo, R: GetRegionInfo> {
    Left(L),
    Right(R),
}

pub(crate) struct Regions<L: GetRegionInfo, R: GetRegionInfo> {
    pub left: Region<L>,
    pub right: Region<R>,
    focused_region: FocusedRegion<L, R>,
}

impl Regions<LeftRegion, RightRegion> {
    pub fn new() -> Self {
        Self {
            left: Region::<LeftRegion>::new(),
            right: Region::<RightRegion>::new(),
            focused_region: FocusedRegion::Left(LeftRegion::Top),
        }
    }

    pub fn set_focus(&mut self, key: KeyCode) -> bool {
        self.focused_region = if key == KeyCode::Char(LeftRegion::Top.get_rect_info().hotkey) {
            FocusedRegion::Left(LeftRegion::Top)
        } else if key == KeyCode::Char(LeftRegion::Middle.get_rect_info().hotkey) {
            FocusedRegion::Left(LeftRegion::Middle)
        } else if key == KeyCode::Char(LeftRegion::Bottom.get_rect_info().hotkey) {
            FocusedRegion::Left(LeftRegion::Bottom)
        } else if key == KeyCode::Char(RightRegion::Top.get_rect_info().hotkey) {
            FocusedRegion::Right(RightRegion::Top)
        } else if key == KeyCode::Char(RightRegion::Bottom.get_rect_info().hotkey) {
            FocusedRegion::Right(RightRegion::Bottom)
        } else {
            return false;
        };
        true
    }

    pub fn draw(&mut self, frame: &mut Frame) {
        // compute rects boundaries and update the regions accordingly
        let outer_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![
                Constraint::Percentage(self.left.info.width_percentage),
                Constraint::Percentage(self.right.info.width_percentage),
            ])
            .split(frame.size());
        let left_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Percentage(LeftRegion::Top.get_rect_info().height_percentage),
                Constraint::Percentage(LeftRegion::Middle.get_rect_info().height_percentage),
                Constraint::Percentage(LeftRegion::Bottom.get_rect_info().height_percentage),
            ])
            .split(outer_layout[0]);
        let right_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Percentage(RightRegion::Top.get_rect_info().height_percentage),
                Constraint::Percentage(RightRegion::Bottom.get_rect_info().height_percentage),
            ])
            .split(outer_layout[1]);
        self.set_rects(left_rects, right_rects);
        // Draw left region
        match self.left.rects {
            Some(_) => {
                for region_variant in LeftRegion::variants() {
                    let is_focused = matches!(&self.focused_region, FocusedRegion::Left(ref lr) if *lr == region_variant);
                    frame.render_widget(
                        self.left.block(&region_variant, is_focused),
                        self.left.rect(&region_variant),
                    );
                }
            }
            None => {}
        }
        // Draw right region
        match self.right.rects {
            Some(_) => {
                for region_variant in RightRegion::variants() {
                    let is_focused = matches!(&self.focused_region, FocusedRegion::Right(ref rr) if *rr == region_variant);
                    frame.render_widget(
                        self.right.block(&region_variant, is_focused),
                        self.right.rect(&region_variant),
                    );
                }
            }
            None => {}
        }
    }

    fn set_rects(&mut self, left_rects: Rc<[Rect]>, right_rects: Rc<[Rect]>) {
        self.left.rects = Some(left_rects);
        self.right.rects = Some(right_rects);
    }
}
